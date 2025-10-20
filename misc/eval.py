import torch
import torch.nn.functional as F


@torch.no_grad()
def test(model, data_loader, max_length, device):
    # switch to evaluate mode
    model.eval()

    dataset = data_loader.dataset
    texts = dataset.text
    num_text = len(texts)
    text_bs = 256

    # Use model's tokenizer if available, otherwise fall back to default
    if hasattr(model, '_tokenizer') and model._tokenizer is not None:
        tokenizer = model._tokenizer
        use_altclip_tokenizer = True
    else:
        from text_utils.tokenizer import tokenize
        use_altclip_tokenizer = False

    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        if use_altclip_tokenizer:
            text_tokens = tokenizer(text, context_length=max_length)
            # Move all tensors to device
            if isinstance(text_tokens, dict):
                text_tokens = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in text_tokens.items()}
            else:
                text_tokens = text_tokens.to(device)
        else:
            text_tokens = tokenize(text, context_length=max_length).to(device)

        text_feat = F.normalize(model.encode_text(text_tokens), dim=-1)
        text_feats.append(text_feat)
    text_feats = torch.cat(text_feats, dim=0)

    image_feats = []
    for image in data_loader:
        image = image.to(device)
        image_feat = F.normalize(model.encode_image(image), dim=-1)
        image_feats.append(image_feat)
    image_feats = torch.cat(image_feats, dim=0)

    sims_matrix = text_feats @ image_feats.t()
    eval_result = metric_eval(sims_matrix, dataset.img2person, dataset.txt2person)

    return eval_result


@torch.no_grad()
def metric_eval(scores_t2i, img2person, txt2person):
    device = scores_t2i.device
    img2person = img2person.to(device)
    txt2person = txt2person.to(device)

    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    real_num = matches.sum(dim=-1)
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long).to(device)
    tmp_cmc /= order
    tmp_cmc *= matches
    AP = tmp_cmc.sum(dim=-1) / real_num
    mAP = AP.mean() * 100.0

    eval_result = {'r1': ir1,
                   'r5': ir5,
                   'r10': ir10,
                   'r_mean': ir_mean,
                   'mAP': mAP.item()
                   }

    return eval_result
