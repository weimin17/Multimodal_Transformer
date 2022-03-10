'''
Visualization. (Consider both positive and negative together)
'''

import pickle
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from PIL import Image


with open('Analysis/bert_analysis_pred_all2.pkl', 'rb') as handle:
    [vis_data_records_ig_l0, tokenlist_top10_l0, tokenlist_bot10_l0, vis_data_records_ig_l1, tokenlist_top10_l1, tokenlist_bot10_l1] = pickle.load(handle)


def save_common_tokens_to_file(tokenlist_top10_l0, file_name):
    '''
    Save tokens to txt file. token along with frequence. 
    '''
    # flatten list to tuple
    flattokenlist_top10_l0 = sum(tokenlist_top10_l0, [])

    # using Counter to find frequency of elements
    frequency = collections.Counter(flattokenlist_top10_l0)
    m = frequency.most_common(400)
    with open('Analysis/{}_2.txt'.format(file_name), 'w') as f:
        for item in m:
            print(item[0], ' ', item[1], file=f)
        f.close()


# save to file. 
save_common_tokens_to_file(tokenlist_top10_l0, file_name = 'pred_tokenlist_top10_l0')



# draw the word cloud
def word_cloud(text_file):
    from wordcloud import WordCloud
    from scipy.ndimage import gaussian_gradient_magnitude


    parrot_color = np.array(Image.open('Analysis/incorrect.png'))
    print('parrot_color', parrot_color.shape)
    parrot_color = parrot_color[:, :, :3]
    # create mask  white is "masked out"
    print('parrot_color', parrot_color.shape)
    parrot_mask = parrot_color.copy()
    parrot_mask[parrot_mask.sum(axis=2) == 0] = 255

    # some finesse: we enforce boundaries between colors so they get less washed out.
    # For that we do some edge detection in the image
    edges = np.mean([gaussian_gradient_magnitude(parrot_color[:, :, i] / 255., 2) for i in range(3)], axis=0)
    parrot_mask[edges > .08] = 255

    # Read the whole text.
    text = ""
    with open('Analysis/{}.txt'.format(text_file), 'r') as f:
        for line in f.readlines():
            a = line.split(' ')
            freq = int( a[3].split('\n')[0] )
            b = [a[0]] * freq
            c = ' '.join(b)
            text = text + c
            # print('a', a, 'b', b, 'c', c, 'text', text)

    # lower max_font_size
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    wordcloud = WordCloud(width=200, height=200, max_font_size=200, max_words=100, mask=parrot_mask, contour_width=3, margin=10, collocations=False, random_state=42, relative_scaling=0.7, mode = "RGB", colormap='gist_ncar').generate(text)
    plt.figure(figsize=(5, 5), dpi=200)
    plt.imshow(wordcloud)
    plt.axis("off")
    # plt.show()
    plt.savefig('Analysis/{}.png'.format(text_file))


    plt.figure(figsize=(5, 5))
    plt.title("Original Image")
    plt.imshow(parrot_color)
    plt.savefig('Analysis/parrot_color.png')


    plt.figure(figsize=(5, 5))
    plt.title("Original Image")
    plt.imshow(parrot_mask)
    plt.savefig('Analysis/parrot_mask.png')


    plt.figure(figsize=(5, 5))
    plt.title("Edge map")
    plt.imshow(edges)
    plt.show()
    plt.savefig('Analysis/edges.png')

word_cloud(text_file='filter_pred_tokenlist_top10_l0_2')







