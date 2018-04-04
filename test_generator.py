import we_panic_utils.nn.processing.frame_processing as fp
import we_panic_utils.nn.data_load as dl
import we_panic_utils.basic_utils.basics as base
from PIL import Image
import matplotlib.pyplot as plt
def main():
    train, test, split = dl.train_test_split_with_csv_support('reg_consolidated', 'NextStartingPoint.csv', 
            'reg_part_out.csv', 'out_test', 'aug_consolidated')

    processor = fp.FrameProcessor().frame_generator(train, "train")

    x = next(processor)[0][0]
    print(type(x[0]))
    base.check_exists_create_if_not("dumb_frames")
    i = 0
    for frame in x: 
        plt.imshow(frame)
        frame *= 255
        img = Image.fromarray(frame.astype('uint8'))
        img.save("dumb_frames/frame{}.png".format(i))
        i+=1
    plt.show()
    print(len(x))

main()
