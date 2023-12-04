import evaluate
import trainer
from PIL import Image
from torchvision import transforms
from utils import visualize

import glob
import random
import torch


TESTDATA_img = "data/WIDER/WIDER_test/images/"
TESTDATA_img_cat = glob.glob(TESTDATA_img + "*")
cat_id = 0
pattern = "*.jpg"


def main():
    args = evaluate.arguments()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    test_loader, templates = evaluate.dataloader(args)
    num_templates = templates.shape[0]

    model = evaluate.get_model(args.checkpoint, num_templates=num_templates)

    test_len = len(test_loader)

    random_img_num = random.choice(range(test_len))
    print(random_img_num)
    print(test_loader.dataset[random_img_num])

    #random_img_cat = random.choice(TESTDATA_img_cat)
    #TESTDATA_img_list = glob.glob(f"{random_img_cat}/{pattern}")

    #print(TESTDATA_img_list[:10])

    #random_img_path = random.choice(TESTDATA_img_list)
    #print('lets test image: ' + random_img_path)

    with torch.no_grad():
        dets = trainer.get_detections(model, test_loader.dataset[random_img_num], templates, test_loader.dataset.rf,
                                      test_loader.dataset.transforms, args.prob_thresh,
                                      args.nms_thresh, device=device)
        
        # convert tensor to PIL image so we can perform resizing
        image_path = f'{TESTDATA_img}{test_loader.dataset[random_img_num][1]}'
        image = Image.open(image_path).convert('RGB')

        visualize.visualize_bboxes(image, dets)


if __name__ == '__main__':
    main()
 