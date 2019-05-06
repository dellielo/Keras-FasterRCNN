import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm


def get_data(input_path, visualise=False):
    all_imgs = []
    classes_count = {}
    class_mapping = {}

    annot_path_ = os.path.join(input_path, 'Tobacc800_Groundtruth_v2.0', 'Tobacc800_Groundtruth_v2.0', 'XMLGroundtruth_v2.0')
    imgs_path = os.path.join(input_path, 'Tobacco800_SinglePage', 'Tobacco800_SinglePage', 'SinglePageTIF')
    imgs_path_list = [os.path.join(imgs_path, s) for s in os.listdir(imgs_path)]
    
    for index, img_path in enumerate(tqdm(imgs_path_list)):
        name_img = os.path.splitext(os.path.basename(img_path))[0]
        annot = os.path.join(annot_path_, name_img + ".xml")
        if os.path.exists(annot):
            exist_flag = False

            # annot_path.set_description("Processing %s" % annot_path.split(os.sep)[-1])
            et = ET.parse(annot)
            element = et.getroot()
            element_doc = element.find('{http://lamp.cfar.umd.edu/GEDI}DL_DOCUMENT')
            filename = element_doc.get('src')
            element_page = element_doc.find('{http://lamp.cfar.umd.edu/GEDI}DL_PAGE')
            element_width = int(element_page.get('width'))
            element_height = int(element_page.get('height'))

            # print(filename, element_width, element_height )
            element_objs = element_page.findall('{http://lamp.cfar.umd.edu/GEDI}DL_ZONE')
            annotation_data = {'filepath': img_path, 'width': element_width,
                                    'height': element_height, 'bboxes': []}
            annotation_data['image_id'] = index
            annotation_data['imageset'] = 'train'
              
            for element_obj in element_objs:
                class_name = element_obj.get('gedi_type')
                
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping) 

                col = element_obj.get('col')
                row = element_obj.get('row')
                width = element_obj.get('width')
                height = element_obj.get('height')
                x1 = int(col)
                y1 = int(row)
                x2 = x1 + int(width)
                y2 = y1 + int(height)
                difficulty = False
                annotation_data['bboxes'].append(
                    {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
            all_imgs.append(annotation_data)

            if visualise:
                img = cv2.imread(annotation_data['filepath'])
                for bbox in annotation_data['bboxes']:
                    cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255), 3)
                print(annotation_data)
                img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                cv2.imshow('img', img )
                cv2.waitKey(0)
    return all_imgs, classes_count, class_mapping

       
if __name__ == "__main__":
    get_data(u"C:\\Users\\Elodie\\Programmation\\DevPython\\sandbox\\ca\\01_data", visualise=True) 