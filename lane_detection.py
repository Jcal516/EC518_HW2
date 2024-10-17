import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import carla
import cv2
import torch

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=120) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=120, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([160,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
        self.state_image_full_old = 0
    

    def front2bev(self, front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.

        input:
            front_view_image)320x240x3

        output:
            bev_image 320x240x3

        '''
        def homography_ipmnorm2g(top_view_region):
            src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
            H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
            return H_ipmnorm2g


        # bev
        bev = np.zeros((240, 320, 3))
        H, W = 240, 320
        top_view_region = np.array([[50, -25], [50, 25], [0, -25], [0, 25]])


        # camera parameters
        cam_xyz = [-1.5, 0, 2.0]
        cam_yaw = 0

        width = 320
        height = 240
        fov = 52

        # camera intrinsic
        focal = width / (2.0 * np.tan(fov * np.pi / 360.0)) 
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0


        # get IPM
        H_g2cam = np.array(carla.Transform(carla.Location(*cam_xyz),carla.Rotation(yaw=cam_yaw),).get_inverse_matrix())
        H_g2cam = np.concatenate([H_g2cam[:3, 0:2], np.expand_dims(H_g2cam[:3, 3],1)], 1)

        trans_mat = np.array([[0,1,0], [0,0,-1], [1,0,0]])
        temp_mat = np.matmul(trans_mat, H_g2cam)
        H_g2im = np.matmul(K, temp_mat)

        H_ipmnorm2g = homography_ipmnorm2g(top_view_region)
        H_ipmnorm2im = np.matmul(H_g2im, H_ipmnorm2g)

        S_im_inv = np.array([[1/np.float(width), 0, 0], [0, 1/np.float(height), 0], [0, 0, 1]])
        M_ipm2im_norm = np.matmul(S_im_inv, H_ipmnorm2im)

        # visualization
        M = torch.zeros(1, 3, 3)
        M[0]=torch.from_numpy(M_ipm2im_norm).type(torch.FloatTensor)

        linear_points_W = torch.linspace(0, 1 - 1/W, W)
        linear_points_H = torch.linspace(0, 1 - 1/H, H)

        base_grid = torch.zeros(H, W, 3)
        base_grid[:, :, 0] = torch.ger(torch.ones(H), linear_points_W)
        base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(W))
        base_grid[:, :, 2] = 1

        grid = torch.matmul(base_grid.view(H * W, 3), M.transpose(1, 2))
        lst = grid[:, :, 2:].squeeze(0).squeeze(1).numpy() >= 0
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])

        x_vals = grid[0,:,0].numpy() * width
        y_vals = grid[0,:,1].numpy() * height

        indicate_x1 = x_vals < width
        indicate_x2 = x_vals > 0

        indicate_y1 = y_vals < height
        indicate_y2 = y_vals > 0

        indicate = (indicate_x1 * indicate_x2 * indicate_y1 * indicate_y2 * lst)*1

        for _i in range(H):
            for _j in range(W):
                _idx = _j + _i*W

                _x = int(x_vals[_idx])
                _y = int(y_vals[_idx])
                _indic = indicate[_idx]

                if _indic == 0:
                    continue

                bev[_i,_j] = front_view_image[_y, _x]

        return np.uint8(bev)


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the image at the front end of the car
        and translate to grey scale

        input:
            state_image_full 320x240x3

        output:
            gray_state_image 320x120x1

        '''
        #print(state_image_full.shape)
        state_image_half = state_image_full[int(state_image_full.shape[0] / 2) - 1:-1,:,:]
        gray_state_image = np.dot(state_image_half, [0.333, 0.333, 0.333]) # evenly sample each channel
        
        return gray_state_image#[::-1] commented out flips it upside down


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 320x120x1

        output:
            gradient_sum 320x120x1

        '''
        percent = 10

        gradient_sum = np.empty([gray_image.shape[0],gray_image.shape[1]])
        for i in range(gray_image.shape[0]):
            gradient_sum[i,:] = np.convolve([-1,0,1], gray_image[i,:], 'same')
        gradient_sum = np.absolute(gradient_sum)
        gradient_sum[gradient_sum < percent * np.max(gradient_sum) / 100] = 0
        gradient_sum[:,0] = 0 # sees the start of the image as an edge, get rid of that
        gradient_sum[:,-1] = 0
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 320x120x1

        output:
            maxima (np.array) 2x Number_maxima

        '''

        np.savetxt("a.csv", gradient_sum, delimiter=",")
        left_m = 0.6503298443658532
        left_b = -1.25496632094531324 + 3
        right_m = -0.6506915142991083
        right_b = 320.32971167369874 - 160 + 1

        argmaxima = np.zeros((gradient_sum.shape[0], 2), dtype='int')
        left = gradient_sum[:, 0 : int(gradient_sum.shape[1] / 2)]
        right = gradient_sum[:, int(gradient_sum.shape[1] / 2) - 1: -1]
        for i in range(gradient_sum.shape[0]):
            index = int(i * left_m + left_b)
            if(index < 0):
                index = 0
            if(index > left.shape[1] - 2):
                index = left.shape[1] - 2
            argmaxima[i,0] = np.argmax(left[i,index:-1]) + index
            if(np.max(left[i,index:-1]) == 0):
                argmaxima[i,0] = -1
            index = int(i * right_m + right_b)
            if(index < 1):
                index = 1
            if(index > right.shape[1]):
                index = right.shape[1]
            argmaxima[i,1] = np.argmax(right[i,0:index]) + right.shape[1] - 1
            if(np.max(right[i,0:index]) == 0):
                argmaxima[i,1] = -1

        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 320x120x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 160:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[320,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def dp_means(self, c_in, left_is_true):
        l = .1
        smallest_cluster = 25
        max_clusters = 10
        c = np.zeros((c_in.shape[0], c_in.shape[1], max_clusters + 1))
        means = np.zeros((max_clusters)) - 1
        sizes = np.zeros((max_clusters))
        ds = np.zeros((max_clusters))
        clusters = 0
        c[:,:,0] = c_in
        c[:,0,0] = (c[:,0,0]-np.min(c[:,0,0]))/(np.max(c[:,0,0])-np.min(c[:,0,0]))
        c[:,1,0] = (c[:,1,0]-np.min(c[:,1,0]))/(np.max(c[:,1,0])-np.min(c[:,1,0]))

        clusters += 1
        c[0,:,clusters] = c[0,:,0]
        c[0,:,0] = [0,0]
        means[clusters - 1] = c[0,0,clusters]
        sizes[clusters - 1] += 1
        ind = 0
        for ind in range(c.shape[0]):
            if(ind == 0):
                continue
            if(clusters == max_clusters):
                print("increase l!")
                break
            a = c[ind,0,0]
            ds = np.zeros((max_clusters)) - 1
            for i in range(clusters):
                ds[i] = np.linalg.norm(a-means[i])
            if(np.min(ds[ds > -1]) > l):
                clusters += 1
                c[ind,:,clusters] = c[ind,:,0]
                c[ind,:,0] = [0,0]
                means[clusters - 1] = c[ind,0,clusters]
                sizes[clusters - 1] += 1
            else:
                argmin = np.argmin(ds[ds > -1])
                c[ind,:,argmin + 1] = c[ind,:,0]
                c[ind,:,0] = [0,0]
                means[argmin] = (means[argmin] * sizes[argmin] + c[ind,0,argmin + 1]) / (sizes[argmin] + 1)
                sizes[argmin] += 1
        
        for i in range(clusters):
            if(sizes[i] < smallest_cluster):
                c[:,:,i + 1] = np.zeros((c.shape[0], c.shape[1]))
        
        thresh = 3
        for i in range(clusters):
            if(not np.sum(c[:,:,i + 1])):
                continue
            mean = np.mean(c[c[:,0,i + 1] > 0,0,i + 1])
            std = np.std(c[c[:,0,i + 1] > 0,0,i + 1])
            c[np.abs(c[:,0,i + 1] - mean) > thresh * std, :, i + 1] = 0
            mean = np.mean(c[c[:,0,i + 1] > 0,1,i + 1])
            std = np.std(c[c[:,0,i + 1] > 0,1,i + 1])
            c[np.abs(c[:,1,i + 1] - mean) > thresh * std, :, i + 1] = 0

        if(left_is_true):
            horiz = c[:,0,:]
            out = c[:,:,np.argmax(np.max(horiz, axis=0))]
            out = out[[out[:,:] != [0,0]][0][:,0],:]
        else:
            horiz = c[:,0,:]
            horiz[horiz[:,:] == 0] = 1
            out = c[:,:,np.argmin(np.min(horiz, axis=0))]
            out = out[[out[:,:] != [1,0]][0][:,0],:]


        out[:,0] = out[:,0]*(np.max(c_in[:,0])-np.min(c_in[:,0])) + np.min(c_in[:,0])
        out[:,1] = out[:,1]*(np.max(c_in[:,1])-np.min(c_in[:,1])) + np.min(c_in[:,1])

        if(out.shape[0] < 4):
            raise error()

        return out


    def lane_detection(self, state_image_full, fig_test):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [320, 240, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        #gray_state = self.cut_gray(state_image_full)
        try:
            bev = self.front2bev(state_image_full) # sometimes fails randomly?
        except:
            bev = self.front2bev(self.state_image_full_old)
        else:
            self.state_image_full_old = state_image_full

        gray_state = np.dot(bev, [0.333, 0.333, 0.333]) # evenly sample each channel

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        #19 0s from bottom

        maxima_img = np.zeros((gradient_sum.shape[0], gradient_sum.shape[1]))
        for i in range(maxima_img.shape[0] - 20):
            if(maxima[i, 0] != -1):
                maxima_img[i + 20, maxima[i, 0]] = 1
            if(maxima[i, 1] != -1):
                maxima_img[i + 20, maxima[i, 1]] = 1
        left = maxima_img[:, 0 : int(maxima_img.shape[1] / 2)]
        right = maxima_img[:, int(maxima_img.shape[1] / 2) - 1: -1]

        a_c = np.array([maxima[:, 0], np.flip(np.arange(0,240,1))])
        b_c = np.array([maxima[:, 1], np.flip(np.arange(0,240,1))])
        a_c = np.delete(a_c, a_c[0, :] < 0, axis=1)
        a_c = a_c.T
        b_c = np.delete(b_c, b_c[0, :] < 0, axis=1)
        b_c = b_c.T
        
        try:
            a_groups = self.dp_means(a_c, True)
        except:
            use_new_left = False
        else:
            use_new_left = True
        try:
            b_groups = self.dp_means(b_c, False)
        except:
            use_new_right = False
        else:
            use_new_right = True

        if(use_new_left and use_new_right):
            '''boundary = np.concatenate((a_groups, b_groups))
            image = np.zeros((240,320))
            for i in range(boundary.shape[0]):
                image[int(boundary[i, 1]), int(boundary[i, 0])] = 1

            left = image[:,0:int(image.shape[1]/2)]
            right = image[:,int(image.shape[1]/2)-1:-1]

            left_b = np.zeros((int(np.sum(left)), 2))
            count = 0
            for i in range(left.shape[0]):
                for j in range(left.shape[1]):
                    if(np.sum(left[i,j])):
                        left_b[count,0] = j
                        left_b[count,1] = i
                        count+=1

            right_b = np.zeros((int(np.sum(right)), 2))
            count = 0
            for i in range(left.shape[0]):
                for j in range(left.shape[1]):
                    if(np.sum(right[i,j])):
                        right_b[count,0] = j + left.shape[1]
                        right_b[count,1] = i
                        count+=1'''
            
            lane_boundary1_points = a_groups
            lane_boundary2_points = b_groups
            
            lane_boundary1, left_u = splprep([lane_boundary1_points[:,0], lane_boundary1_points[:,1]], s=10000)
            lane_boundary2, left_u = splprep([lane_boundary2_points[:,0], lane_boundary2_points[:,1]], s=10000)
                
            #else:
            #    lane_boundary1 = self.lane_boundary1_old
            #    lane_boundary2 = self.lane_boundary2_old
            ################

            self.lane_boundary1_old = lane_boundary1
            self.lane_boundary2_old = lane_boundary2

            # testing
            plt.figure("test")
            plt.scatter(lane_boundary1_points[:,0],lane_boundary1_points[:,1])
            plt.scatter(lane_boundary2_points[:,0],lane_boundary2_points[:,1])
        
        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        plt.figure("test")
        plt.gcf().clear()
        plt.imshow(image)#maxima_img)
        plt.xlim(0, 320)
        plt.ylim(0, 240)
        plt.axis('off')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig_test.canvas.flush_events()

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        
        plt.figure("gt")
        plt.gcf().clear()
        plt.imshow(state_image_full)#[::-1])
        if(self.lane_boundary1_old != 0):
            lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
            plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+320-self.cut_size, linewidth=5, color='orange')
        if(self.lane_boundary2_old != 0):
            lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
            plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+320-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+320-self.cut_size, color='white')

        plt.axis('off')
        #plt.xlim((-0.5,95.5))
        #plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
