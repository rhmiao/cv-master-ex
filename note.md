# block 的生成

主要想法是通过宽度优先搜索算法(BFS)生成离中心点拓扑等距的block

1. 生成连通图

   ```python
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
   ```

   

2. 通过BFS生成blocks

   ```python
   while(last_insert_ids):
       insert_ids=last_insert_ids.pop()
       for insert_id in insert_ids:
           if(insert_id in image_id_set):
               continue
           n_divided_block=n_divided_block+1
           blocks[n_divided_block]=[]
           bfs_insert_ids,id_set = bfs(insert_id)
           blocks[n_divided_block]=list(id_set)
           blocks[n_divided_block].sort()
   
           image_id_set=image_id_set.union(id_set)# update ids
   
           ind=0
           # update bfs_insert_ids for adding in queue
           while ind <len(bfs_insert_ids):
               bfs_insert_id=bfs_insert_ids[ind]
               if(bfs_insert_id in image_id_set):
                   bfs_insert_ids.pop(ind)# delete the known id in image_id_set
                   ind-=1
                   for new_id in connect_graph[bfs_insert_id]:# find the next image_ids
                       if(new_id not in image_id_set):
                           ind+=1
                           bfs_insert_ids.insert(ind,new_id)
               ind+=1
           if(bfs_insert_ids):last_insert_ids.insert(0,bfs_insert_ids)
   ```

   

3. 根据生成的blocks生成selectors

```python
selectors={};
for num in range(len(blocks)):
    selector_name = 'kitti-sub'+format(str(num), '0>2s')
    selectors[selector_name]={
        "sample_stride": 1,
        "train": lambda length,block=blocks[num]: [block[i] for i in range(0,len(block), 1) if i % 10],
        "val": lambda length,block=blocks[num]: [block[i] for i in range(0,len(block), 10)],
        "test": lambda length,block=blocks[num]: [block[i] for i in range(0,len(block), 1)],
    }
```

代码见nerf/atlantic_datasets/block_selector.py

# block的融合

主要思路是：对于目标图像，根据每一个block里面图像离目标图像的距离和方向差异进行加权，权重计算为
$$
W_i=\sum_j(\frac{cos(yaw(i,j))}{1+distance(i,j)})
$$

1. 利用.\test_nerf.py遍历每一个block生成prediction的图

   ```python
       for subcase in range(len(selectors)):
           opt.data.selector='kitti-sub'+format(str(subcase), '0>2s')
           model = NeRFNetwork(
   	    encoding="hashgrid"
   	    if not opt.extrinsic.optimize_extrinsics
   	    else "annealable_hashgrid",
               bound=opt.bound,
               cuda_ray=opt.cuda_ray,
               **opt.network,
           )
           print(model)
           print('starint test kitti-sub'+format(str(subcase), '0>2s'));
           # Test
           metrics = Metric(opt.metrics)
   
           _, _, test_dataset = Dataset(
               bound=opt.bound,
               **opt.data,
           )
           test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
   
           if opt.renderer.z_far <= 0:
               opt.renderer.z_far = float(test_dataset.depth_scale)
   
           sampler = Sampler(**opt.sampler, class_colors=test_dataset.class_colors)
   
           trainer = Trainer(
               name='kitti-sub'+format(str(subcase), '0>2s'),
               conf=opt.renderer,
               model=model,
               metrics=metrics,
               workspace=workspace,
               fp16=opt.fp16,
               sampler=sampler,
               use_checkpoint="latest",
               depth_scale=test_dataset.depth_scale,
           )
   
           trainer.test(test_loader, alpha_premultiplied=opt.test.alpha_premultiplied)
   ```

   

2. 利用.\merge_images.py遍历每一张图并计算权重，然后进行融合并输出

   ```python
   def get_weight(reference_id,target_block,img_poses):
       reference_pose = img_poses[reference_id]
       weight_sum=0
       for target_id in target_block:
           target_pose = img_poses[target_id]
           dis = np.linalg.norm(target_pose[:3]-reference_pose[:3],2)
           target_R=scipy.spatial.transform.Rotation.from_quat(target_pose[[4, 5, 6, 3]]).as_matrix()
           reference_R=scipy.spatial.transform.Rotation.from_quat(reference_pose[[4, 5, 6, 3]]).as_matrix()
           delta_R=target_R..transpose()*reference_R
           cosine_yaw=(delta_R.trace()-1)/2
           weight_sum += cosine_yaw/(1+dis)
       return weight_sum
   
       for i in range(n_images):
           img_path_list=[]
           merge_weights=[]
           weight_sum=0
           merge_img=None
           for j in range(len(selectors)):
               name='kitti-sub'+format(str(j), '0>2s')
               if(i in selectors[name]['test']):
                   index_i = selectors[name]['test'].index(i)
                   img_path=os.path.join(workspace, "results", f"{name}_{index_i:04d}.png")
                   img_path_list.append(img_path)
                   #calculate merge weight
                   weight = get_weight(i,selectors[name]['test'],poses)
                   merge_weights.append(weight)
                   img = cv2.imread(img_path).astype("float32")
                   
                   weight_sum += weight
                   if(merge_img is None):
                       merge_img = img
                   else:
                       merge_img += weight * img
           #merge image
           merge_img = (merge_img / weight_sum).astype("uint8")   
           save_path = os.path.join(save_path_root, f"{i:04d}.png")
           cv2.imwrite(save_path, merge_img)
   ```

   