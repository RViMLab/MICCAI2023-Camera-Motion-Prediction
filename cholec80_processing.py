import os
import cv2
import numpy as np
import yaml

from utils.io import scan2df, save_yaml, load_yaml
from utils import endoscopy

if __name__ == '__main__':
    servers = load_yaml('configs/servers.yml')
    server = 'rvim_server'

    prefix = servers[server]['database']['location']

    # scan through data
    folder = os.path.join(prefix, 'cholec80/videos')
    print('Scanning: ', folder)

    df = scan2df(folder, postfix='.mp4')

    print('Found:\n')
    print(df)

    # create object to save found transforms
    database =  {
        'databases': [{
            'name': 'cholec80',
            'prefix': 'cholec80',
            'test': False,
            'transforms': [],
            'videos': {
                'files': [],
                'prefix': 'videos'
    }}]}

    # find center, radius and rectangle crop
    visualize = False
    stable_steady_count = 50  # center is defined stable after 50 consecutive images have the same center
    stable_zoom_count = 1000  # zoom is considered stable, after 1000 consecutive images are zoomed

    steady_tol = 1.0
    zoom_tol = 0.99

    # offset crop to remove black borders
    offset = 5

    # output
    out_prefix = os.path.join(os.getcwd(), 'configs')
    out_file = 'cholec80_transforms.yml'

    dsize = [640, 480] # opencv convention for resize output


    for idx, row in df.iterrows():
        ib = endoscopy.ImageBuffer(buffer_size=50)

        vc = cv2.VideoCapture(os.path.join(row.folder, row.file))

        print('Processing file: {} with index: {}'.format(row.file, idx))

        # steady center check
        steady_count = 0
        prev_center, prev_radius = np.array([0, 0], dtype=np.int), None

        # zoom check
        zoom_count = 0

        # found flags
        center_found = False
        zoom = False

        while vc.isOpened():

            _, img = vc.read()
            if img is None:
                break

            img = img[offset:-offset, offset:-offset]
            ib.appendBuffer(img)

            avg_binary = ib.binaryAvg(th=20)

            center, radius = endoscopy.ransacBoundaryCircle(avg_binary, th=10, fit='numeric', n_pts=100, n_iter=10)
            top_left, shape = endoscopy.boundaryRectangle(avg_binary, th=200)

            # check for circle fit
            if radius is not None:
                # find max inner rectangle
                inner_top_left, inner_shape = endoscopy.maxRectangleInCircle(img.shape, center, radius)
                inner_top_left, inner_shape = inner_top_left.astype(np.int), tuple(map(int, inner_shape))
        
                center, radius = center.astype(np.int), int(radius)
                top_left, shape = top_left.astype(np.int), tuple(map(int, shape))

                if steady_count == 0:
                    prev_center, prev_radius = center, radius
                    steady_count += 1
                else:
                    if np.isclose(prev_center, center, atol=steady_tol).all():
                        prev_center, prev_radius = center, radius
                        steady_count += 1

                        if steady_count >= stable_steady_count + 1:
                            center_found = True
                            break
                    else:
                        prev_center, prev_radius = np.array([0, 0], dtype=np.int), None
                        steady_count = 0

                if visualize:
                    cv2.circle(img, (center[1], center[0]), radius, (0, 255, 255))
                    cv2.circle(img, (center[1], center[0]), 2, (0, 255, 255))
                    cv2.rectangle(img, (top_left[1], top_left[0]), (top_left[1]+shape[1], top_left[0]+shape[0]), (255, 0, 255))
                    cv2.rectangle(
                        img, 
                        (inner_top_left[1], inner_top_left[0]), 
                        (inner_top_left[1]+inner_shape[1], inner_top_left[0]+inner_shape[0]), 
                        (255, 255, 0)
                    )

            # check for zoom
            zoomed, confidence = endoscopy.isZoomed(avg_binary, th=zoom_tol)
            if zoomed:
                zoom_count += 1
            else:
                zoom_count = 0
            if zoom_count > stable_zoom_count:
                zoom = True
                break

            if visualize:
                cv2.imshow('avg_binary', avg_binary)
                cv2.imshow('img', img)
                cv2.waitKey(1)
            
            print('\rSteady count: {}, zoom count: {}'.format(steady_count, zoom_count), end='')

        vc.release() 
        print('\nCenter found: {}, zoom: {}'.format(center_found, zoom))
        print('Center found at: ', center, '\n')

        # save results, remember to add offset again
        if center_found and not zoom:
            database['databases'][0]['transforms'].append(
                [{'Crop': {'top_left_corner': [inner_top_left.item(0) + offset, inner_top_left.item(1) + offset], 'shape': [inner_shape[0], inner_shape[1]]}}, {'Resize': {'dsize': dsize}}]
            )
            database['databases'][0]['videos']['files'].append(row.file)

        if visualize:
            if radius is not None:
                cv2.circle(img, (center[1], center[0]), radius, (0, 255, 255))
                cv2.circle(img, (center[1], center[0]), 2, (0, 255, 255))
                cv2.rectangle(img, (top_left[1], top_left[0]), (top_left[1]+shape[1], top_left[0]+shape[0]), (255, 0, 255))
                cv2.rectangle(
                    img, 
                    (inner_top_left[1], inner_top_left[0]), 
                    (inner_top_left[1]+inner_shape[1], inner_top_left[0]+inner_shape[0]), 
                    (255, 255, 0)
                )

            cv2.imshow('avg_binary', avg_binary)
            cv2.imshow('img', img)
            cv2.waitKey()

    # save resulting yaml file
    save_yaml(os.path.join(out_prefix, out_file), database)
