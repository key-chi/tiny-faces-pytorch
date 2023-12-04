import evaluate
import trainer
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

    random_img = random.choice(list(test_loader))
    print(random_img)

    #random_img_cat = random.choice(TESTDATA_img_cat)
    #TESTDATA_img_list = glob.glob(f"{random_img_cat}/{pattern}")

    #print(TESTDATA_img_list[:10])

    #random_img_path = random.choice(TESTDATA_img_list)
    #print('lets test image: ' + random_img_path)

    with torch.no_grad():
        dets = trainer.get_detections(model, random_img, templates, test_loader.dataset.rf,
                                      test_loader.dataset.transforms, args.prob_thresh,
                                      args.nms_thresh, device=device)
        
        visualize.visualize_bboxes()


if __name__ == '__main__':
    main()
 