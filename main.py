import os
import random
import time
from pathlib import Path

import torch

from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model import clip_vitb
from options import get_args

from model.altclip_adapter import build_altclip_clip


def run(config):
    print(config)

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    # Add num_classes to config
    config.model.num_classes = num_classes

    # Initialize meters for tracking losses
    meters = {
        "loss": AverageMeter(),
        "nitc_loss": AverageMeter(),
        "ss_loss": AverageMeter(),
        "citc_loss": AverageMeter(),
        "ritc_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "id_loss": AverageMeter(),
    }
    best_rank_1 = 0.0
    best_epoch = 0

    # model
    model, tokenizer, processor = build_altclip_clip(config, model_name=config.model.checkpoint)
    model.to(config.device)

    # Load fine-tuned checkpoint if exists
    checkpoint_path = os.path.join(config.model.saved_path, 'checkpoint_best.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading fine-tuned checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        load_result = model.load_state_dict(ckpt['model'], strict=False)
        print(f"Checkpoint loaded: {load_result}")
    else:
        print("No fine-tuned checkpoint found, using pretrained AltCLIP model")

    if is_using_distributed():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # schedule
    config.schedule.niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config)

    # optimizer
    optimizer = build_optimizer(config, model)

    # train
    it = 0
    scaler = torch.cuda.amp.GradScaler()

    # For ETA calculation
    epoch_times = []
    training_start_time = time.time()

    for epoch in range(config.schedule.epoch):
        print()
        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        epoch_start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for i, batch in enumerate(train_loader):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']

            if epoch == 0:
                alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
            else:
                alpha = config.model.softlabel_ratio

            if config.experiment.mixgen:
                if random.random() < config.experiment.mixgen_p:
                    import model.mixgen as mg
                    if config.experiment.mixgen_type == 'cat':
                        mixgen_func = mg.concatgen
                    else:
                        mixgen_func = mg.mixgen
                    img, cap = mixgen_func(batch['image'], batch['caption'],
                                           num=int(config.experiment.mixgen_ratio * len(batch['caption'])))
                    batch.update({
                        'image': img,
                        'caption': cap,
                    })

            with torch.autocast(device_type='cuda'):
                ret = model(batch, alpha)
                loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['image'].shape[0]
            meters['loss'].update(loss.item(), batch_size)
            meters['nitc_loss'].update(ret.get('nitc_loss', 0), batch_size)
            meters['ss_loss'].update(ret.get('ss_loss', 0), batch_size)
            meters['citc_loss'].update(ret.get('citc_loss', 0), batch_size)
            meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:
                # Calculate ETA
                elapsed_time = time.time() - epoch_start_time
                batches_done = i + 1
                batches_left_in_epoch = len(train_loader) - batches_done
                time_per_batch = elapsed_time / batches_done
                eta_epoch = time_per_batch * batches_left_in_epoch

                # Calculate total ETA based on average epoch time
                if epoch_times:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                    epochs_left = config.schedule.epoch - epoch - 1
                    total_eta = eta_epoch + (avg_epoch_time * epochs_left)
                else:
                    # First epoch: estimate based on current progress
                    estimated_epoch_time = time_per_batch * len(train_loader)
                    epochs_left = config.schedule.epoch - epoch - 1
                    total_eta = eta_epoch + (estimated_epoch_time * epochs_left)

                # Format ETA
                def format_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

                info_str = f"Epoch[{epoch + 1}/{config.schedule.epoch}] Iteration[{i + 1}/{len(train_loader)}]"
                # log loss
                for k, v in meters.items():
                    if v.val != 0:
                        info_str += f", {k}: {v.val:.4f}"
                info_str += f", Base Lr: {param_group['lr']:.2e}"
                info_str += f", ETA: {format_time(eta_epoch)} (Total: {format_time(total_eta)})"
                print(info_str)

        if is_master():
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            time_per_batch = epoch_time / len(train_loader)

            # Calculate remaining time
            if len(epoch_times) > 0:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                epochs_remaining = config.schedule.epoch - epoch - 1
                total_remaining_time = avg_epoch_time * epochs_remaining

                def format_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

                print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s] Total ETA: {}"
                      .format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch,
                              format_time(total_remaining_time)))

            # Get the correct model reference (handle distributed training)
            test_model = model.module if is_using_distributed() else model

            # Evaluate on multiple languages
            text_length = config.experiment.text_length
            test_loaders = dataloader['test_loaders']

            print("\n" + "="*50)
            print(f"Evaluation Results for Epoch {epoch + 1}")
            print("="*50)

            all_results = {}
            for lan, test_loader in test_loaders.items():
                eval_result = test(test_model, test_loader, text_length, config.device)
                rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
                all_results[lan] = {'r1': rank_1, 'r5': rank_5, 'r10': rank_10, 'mAP': map}
                print(f'[{lan.upper()}] Acc@1 {rank_1:.5f} Acc@5 {rank_5:.5f} Acc@10 {rank_10:.5f} mAP {map:.5f}')

            # Calculate average metrics across all languages
            avg_rank_1 = sum([r['r1'] for r in all_results.values()]) / len(all_results)
            avg_rank_5 = sum([r['r5'] for r in all_results.values()]) / len(all_results)
            avg_rank_10 = sum([r['r10'] for r in all_results.values()]) / len(all_results)
            avg_map = sum([r['mAP'] for r in all_results.values()]) / len(all_results)
            print(f'[AVG] Acc@1 {avg_rank_1:.5f} Acc@5 {avg_rank_5:.5f} Acc@10 {avg_rank_10:.5f} mAP {avg_map:.5f}')
            print("="*50 + "\n")

            torch.cuda.empty_cache()
            # Use average R@1 as the criterion for best model
            if best_rank_1 < avg_rank_1:
                best_rank_1 = avg_rank_1
                best_epoch = epoch

                # Get the correct model state dict
                model_state = test_model.state_dict()

                save_obj = {
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }
                torch.save(save_obj, os.path.join(config.model.saved_path, 'checkpoint_best.pth'))

    # Print final statistics
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    secs = int(total_training_time % 60)
    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{secs:02d}")
    print(f"Best Acc@1: {best_rank_1:.5f} at epoch {best_epoch + 1}")
    print("="*50)


if __name__ == '__main__':
    config_path = 'config/config.yaml'

    args = get_args()
    if args.simplified:
        config_path = 'config/s.config.yaml'
    config = parse_config(config_path)

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)

    init_distributed_mode(config)

    set_seed(config)

    run(config)
