import cv2
import numpy as np
import os



def sad_calculation(alpha_path, sal_path):
    alpha = cv2.imread(alpha_path)[:, :, 0] / 255.
    # origin_pred_mattes = cv2.imread(sal_path, cv2.IMREAD_UNCHANGED)[:, :, 3] / 255.

    origin_pred_mattes = cv2.imread(sal_path, 0)/ 255.

    #done to avoid cuda out of error issue
    alpha = cv2.resize(alpha, (500,500))
    origin_pred_mattes = cv2.resize(origin_pred_mattes, (500, 500))

    assert(alpha.shape == origin_pred_mattes.shape)

    mse_diff = ((origin_pred_mattes - alpha) ** 2).sum()
    sad_diff = np.abs(origin_pred_mattes - alpha).sum()

    print(sad_diff)

    return sad_diff, mse_diff


def missing_images(dir1, dir2):
    dir1_imgs = os.listdir(dir1)
    dir2_imgs = os.listdir(dir2)
    dir2_imgs = [img.replace('-depositphotos-bgremover', '') for img in dir2_imgs]
    missing  = [img for img in dir1_imgs if img not in dir2_imgs]
    return missing


def merge_masks(dir1, dir2):
    result_dir = '/home/kartik/Downloads/detic_results/version4/avg/masks'
    os.makedirs(result_dir, exist_ok=True)
    for f_path in os.listdir(dir1):
        path_1 = os.path.join(dir1, f_path)
        path_2 = os.path.join(dir2, f_path.replace('.jpg', '_sal_fuse.png'))
        mask1 = cv2.imread(path_1, 0)
        mask2 = cv2.imread(path_2, 0)
        mask = np.concatenate((mask1.reshape(1, mask1.shape[0], mask1.shape[1]),
                mask2.reshape(1, mask2.shape[0], mask2.shape[1]) ), axis=0)
        mask = np.mean(mask, axis=0)
        cv2.imwrite(os.path.join(result_dir, f_path), mask)
    return result_dir

if __name__ == '__main__':
    mse_diffs = 0
    sad_diffs = 0

    # missing = missing_images('/home/kartik/Documents/work/datasets/test-dataset-50/api-results-v15', '/home/kartik/Documents/work/datasets/test-dataset-50/deposit-photos-results')

    alpha_dir = '/home/ubuntu/kartik/datasets/test-dataset/annotations'
    sal_dir = '/home/ubuntu/kartik/LDF/eval/maps/LDF/test-800'

    # result_dir = merge_masks(sal_dir, '/home/kartik/Downloads/detic_results/v15_asp_resized')
    # sal_dir = result_dir

    file_paths = os.listdir(sal_dir)

    cnt = len(file_paths)

    

    for idx, sal_path in enumerate(file_paths):
        # sal_path = alpha_path.replace('.jpg', '_sal_fuse.png')
        # sal_path = alpha_path
        print(idx)
        # sal_path = alpha_path.replace('.jpg', '.png')
        # alpha_path = sal_path.replace('-depositphotos-bgremover.png', '.jpg')
        # alpha_path = sal_path.replace('_sal_fuse.png', '.jpg', )
        alpha_path = os.path.join(alpha_dir, sal_path.replace('.png', '.jpg'))
        sal_path = os.path.join(sal_dir, sal_path)
        sad_diff, mse_diff = sad_calculation(alpha_path, sal_path)
        mse_diffs += mse_diff
        sad_diffs += sad_diff

    print('final sad is ')
    print(sad_diffs/cnt)
    pass