from .text_generator import DocumentProcessor, WordGenerator, Writer
import numpy as np

path = ['book8.txt']
def oktop():
    dp = DocumentProcessor(document_path = path)
    dp.process_document(vec_num=10, count_num=0)
    x, y = dp.get_training_data(time_step=5)
    x_shape = (x.shape[1], x.shape[2])
    y_shape = y.shape[2]

    print(x_shape)
    print(y_shape)

    wg = WordGenerator(x_shape, y_shape)
    wg.train(x, y, batch_n=10, epochs_n=1)

    # dp.save()
    # wg.save()

    writer = Writer(dp, wg, time_step=5)

    train = writer.write(0, dp.corpus[:4], number_of_word=100)
    return train
