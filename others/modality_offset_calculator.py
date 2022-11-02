import pickle, torch


DBG = False


def get_centers_info():
    with open('coco_clip_embeddings/oscar_split_RN50x4_train_with_text_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        images_embeddings = data['clip_embedding'][:100 if DBG else 20000].float()
        images_embeddings /= torch.norm(images_embeddings, dim=1, keepdim=True)

        text_embeddings = data['clip_embedding_text_dave'][:100 if DBG else 20000].float()
        text_embeddings /= torch.norm(text_embeddings, dim=1, keepdim=True)

    center_image = images_embeddings.mean(dim=0, keepdim=True)
    center_text = text_embeddings.mean(dim=0, keepdim=True)
    offset_to_add_in_inference = center_text - center_image
    offset_to_add_in_training = center_image - center_text

    #stats
    dist = (text_embeddings - images_embeddings)
    print(f'Offset analysis: L2 norm={dist.mean(dim=0).norm():.2f}, Mean={dist.abs().mean(dim=0).mean():.2f}, Max={dist.mean(dim=0).abs().max():.2f}, Min={dist.mean(dim=0).abs().min():.2f}')
    print(f'STD of Offset analysis: L2 norm={dist.std(dim=0).norm():.2f}, Mean={dist.std(dim=0).mean():.2f}, Max={dist.std(dim=0).max():.2f}, Min={dist.std(dim=0).min():.2f}')
    return center_text, center_image, offset_to_add_in_training, offset_to_add_in_inference


def get_variance_info():
    with open('coco_clip_embeddings/oscar_split_RN50x4_train_with_text_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        images_embeddings = data['clip_embedding'][:100 if DBG else 200000].float()
        images_embeddings /= torch.norm(images_embeddings, dim=1, keepdim=True)

        text_embeddings = data['clip_embedding_text_dave'][:100 if DBG else 200000].float()
        text_embeddings /= torch.norm(text_embeddings, dim=1, keepdim=True)
    diff = images_embeddings - text_embeddings
    diff_std = diff.std(dim=0)
    print(f'STD norm of diff ={diff_std.norm()}')

    image_std = images_embeddings.std(dim=0)
    print(f'STD norm of image embeddings ={image_std.norm()}')

    text_std = text_embeddings.std(dim=0)
    print(f'STD norm of text embeddings ={text_std.norm()}')


def save_centers_info_to_pickle():
    center_text, center_image, offset_to_add_in_training, offset_to_add_in_inference = get_centers_info()
    with open('CLIP_embeddings_centers_info.pkl', 'wb') as f:
        pickle.dump({
            'center_text': center_text,
            'center_image': center_image,
            'offset_to_add_in_training': offset_to_add_in_training,
            'offset_to_add_in_inference': offset_to_add_in_inference,
        }, f)
    print(f'norm of diff ={torch.norm(offset_to_add_in_inference)}')
    print('saved centers info to pickle successfully')


def get_precalculated_centers():
    with open('CLIP_embeddings_centers_info.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # save_centers_info_to_pickle()
    # centers = get_precalculated_centers()
    get_centers_info()
    a=0
