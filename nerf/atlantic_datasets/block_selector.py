import numpy as np
import cv2
import os
from dataset_kitti_odometry import KITTI_Odometry

scene = "00"
evalset = KITTI_Odometry(scene=scene)
scene_info=evalset._build_dataset()

data_info=scene_info['00'];
poses=data_info['poses_left'];
n_images=data_info['n_frames'];
'''
testing 01
'''
#print(poses[0])
#print(data_info['n_frames'])
'''
building block with BFS
'''
## generate connected graph
mean_dis=0
pose_nodes={}
n_nodes=0

for i in range(n_images-1):
    dis=np.linalg.norm(poses[i+1][:3]-poses[i][:3],2);
    mean_dis = mean_dis + dis
    #if(dis<0.2):
    #    pose_nodes[n_nodes]
    #else:
    #    n_nodes=n_nodes+1
    #print(i,dis)
mean_dis = mean_dis/(n_images-1);
print('mean dis is:',mean_dis);
max_connect_dis = 2*mean_dis;
## generate sub-graph for each block
connect_graph={}
def node_block_dist(frame_id,connect_list):
    for target_id in connect_list:
        if abs(target_id-frame_id)<50:
            return False;
    return True

for i in range(n_images):
    connect_graph[i]=[]
    if(i<n_images-1):
        connect_graph[i].append(i+1)
    if(i>0):
        connect_graph[i].append(i-1)
    for j in range(i+50,n_images):#other connections should be in (-inf,n-50] u [n+50,inf)
        dis=np.linalg.norm(poses[i][:3]-poses[j][:3],2);
        if(dis<max_connect_dis):# add the id which the distance is less than setting max connect distance
            if(node_block_dist(j,connect_graph[i])):connect_graph[i].append(j)
    for j in range(i-50):#other connections should be in (-inf,n-50] u [n+50,inf)
        dis=np.linalg.norm(poses[i][:3]-poses[j][:3],2);
        if(dis<max_connect_dis):# add the id which the distance is less than setting max connect distance
            if(node_block_dist(j,connect_graph[i])):connect_graph[i].append(j)
#for i in range(n_images):
#    print("i:",i,connect_graph[i])
            
## generate blocks 
# parameters
n_max_block_images=100
n_divided_images=0
n_divided_block=-1
# output blocks
blocks={}
image_id_set = set()
last_image_id_set = set()
# BFS
def bfs(start_id):
    queue = []
    id_set = set() 
    queue.insert(0,[start_id])
    id_set.add(start_id)
    n_count=1
    last_ids=[]
    while (n_count<n_max_block_images and queue):
        cur_id_list = queue.pop()                    # pop element
        if(cur_id_list):
            last_ids=[]
        sub_queue=[]
        for cur_id in cur_id_list:
            for next_id in connect_graph[cur_id]:   # traversing adjacent nodes of elements
                if next_id not in id_set:           # If the adjacent node has not been queued, join the queue and register
                    id_set.add(next_id)
                    sub_queue.insert(0,next_id)
                    n_count=n_count+1
                    last_ids.append(next_id)
        if(sub_queue):
            queue.insert(0,sub_queue)
    return last_ids,id_set

last_insert_ids=[[0]]
'''
blocks[0]=[]
bfs_insert_ids,id_set = bfs(100,0)
print("bfs_insert_ids:",bfs_insert_ids)
print("id_set:",id_set)  
'''         

while(last_insert_ids):
    insert_ids=last_insert_ids.pop()
    #print("insert_ids:",insert_ids)
    for insert_id in insert_ids:
        #print("insert_id:",insert_id)
        #print("insert_id in:",insert_id in image_id_set)
        if(insert_id in image_id_set):
            continue
        n_divided_block=n_divided_block+1
        blocks[n_divided_block]=[]
        bfs_insert_ids,id_set = bfs(insert_id)
        blocks[n_divided_block]=list(id_set)
        blocks[n_divided_block].sort()
        #print(f"{n_divided_block}:{blocks[n_divided_block]}")
        #print("last_insert_ids:",last_insert_ids)
        image_id_set=image_id_set.union(id_set)
        #print("before delete bfs_insert_ids:",bfs_insert_ids)
        ind=0
        while ind <len(bfs_insert_ids):
            bfs_insert_id=bfs_insert_ids[ind]
            if(bfs_insert_id in image_id_set):
                bfs_insert_ids.pop(ind)
                ind-=1
                for new_id in connect_graph[bfs_insert_id]:
                    if(new_id not in image_id_set):
                        ind+=1
                        bfs_insert_ids.insert(ind,new_id)
            ind+=1
        #print("after delete bfs_insert_ids:",bfs_insert_ids)
        if(bfs_insert_ids):last_insert_ids.insert(0,bfs_insert_ids)
        #print("id_set:",id_set)
    #print("all_id_set:",all_id_set)

'''
#output debug
print("n_divided_block:",n_divided_block)
print("image_id_set:",len(image_id_set),n_images)
for i in range(n_divided_block):
    blocks[i].sort()
    print(f"{i}:{blocks[i]}")
'''
#generate selectors
selectors={};
for num in range(len(blocks)):
    selector_name = 'kitti-sub'+format(str(num), '0>2s')
    selectors[selector_name]={
        "sample_stride": 1,
        "train": lambda length,block=blocks[num]: [block[i] for i in range(0,len(block), 1) if i % 10],
        "val": lambda length,block=blocks[num]: [block[i] for i in range(0,len(block), 10)],
        "test": lambda length,block=blocks[num]: [block[i] for i in range(0,len(block), 1)],
    }
'''
#testing 02
'''    
print(selectors['kitti-sub01']['train'](2000))
print(selectors['kitti-sub01']['val'](2000))