import pandas as pd
import requests
import glob
import os
import click


def urls_to_download(image_classes):
    # cat class-descriptions-boxable.csv  | grep -i door
    # cat   train-annotations-bbox.csv  | grep /m/02dgv | awk -F',' '{print $1}' | sort | uniq  | wc -l
    print('urls_to_download: image_classes:', image_classes)
    class_descriptions = pd.read_csv('data/class-descriptions-boxable.csv', names=['key', 'name'])
    print(class_descriptions.head())
    
    class_descriptions = class_descriptions[class_descriptions['name'].str.lower().isin(image_classes)]
    print(class_descriptions)
    
    # train-annotations-bbox.csv
    # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
    # 000002b66c9c498e,xclick,/m/01g317,1,0.012500,0.195312,0.148438,0.587500,0,1,0,0,0
    annotations = pd.read_csv('data/train-annotations-bbox.csv')
    
    filtetered_annotations = annotations[annotations['LabelName'].isin(class_descriptions['key'])]
    print(filtetered_annotations.head())
    
    image_ids = pd.unique(filtetered_annotations['ImageID'])
    print(len(image_ids))
    print(image_ids[:10])
    
    
    images = pd.read_csv('data/train-images-boxable.csv')
    
    urls = images[images['ImageID'].isin(image_ids)]
    return urls


def download_images(output_dir, pd_urls):
    for index, row  in pd_urls.iterrows():
        print("download images in {}".format(output_dir))    
        name = row['ImageID']
        url = row['OriginalURL']
        try:
        
            if os.path.exists('{}/{}.jpg'.format(output_dir, name)):
                print("file already downloaded {}".format(url))
                continue
        
            print("downloading {} from {}".format(name, url))
        
            data = requests.get(url, verify=False, timeout=5).content
            with open('{}/{}.jpg'.format(output_dir, name), 'wb') as wf:
                wf.write(data)
        except Exception as ex:
            print("failed to download {} with error {}".format(url, ex))


@click.command()
@click.option('--output_dir', default='output', help='output directory')
@click.option('--image_classes', type=str, multiple=True, default=('door', 'window', 'billboard'))
def cli(output_dir, image_classes):
    click.echo("Downloading images to {}".format(output_dir))
    download_images(output_dir, urls_to_download(image_classes))

if __name__ == '__main__':
    cli()





