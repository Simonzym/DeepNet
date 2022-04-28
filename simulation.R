library(plot.matrix)
library(oro.nifti)
library(dplyr)
library(stringr)
setwd('Y:/PhD/Fall 2020/RA')
cur_path = getwd()

#area centered by c1 and c2 are related to label; area around c3 are noise
c1 = c(10, 10, 10)
c2 = c(20, 20, 20)
c3 = c(10, 10, 20)
#four visits available, label from 5th visit available
num.vis = 4
half.vis = 2

#grids information
g = expand.grid(1:30, 1:30, 1:30)
g$d1 = sqrt ((g$Var1-c1[1])^2 + (g$Var2-c1[2])^2 + (g$Var3 - c1[3])^2)
g$d2 = sqrt ((g$Var1-c2[1])^2 + (g$Var2-c2[2])^2 + (g$Var3 - c2[3])^2)
g$d3 = sqrt ((g$Var1-c3[1])^2 + (g$Var2-c3[2])^2 + (g$Var3 - c3[3])^2)

#create normal control
create.nc = function(start, end, folder, set = 'train', noise, p = 0.3){
  
  new.df = data.frame()
  for(id in start:end){
    #if generate noisy area
    gen.noise = runif(1)<=p
    fold.name = paste0(cur_path, '/Data/', folder, '/', set, '/', 
                       str_pad(id, 3, pad = '0'))
    dir.create(fold.name)
    #base image: all pixel values around 2.5
    bl = array(rnorm(30^3, 2.5, 0.05), c(30, 30, 30))
    for(vis in 1:num.vis){
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = vis, ad = 0)
      new.df = rbind(new.df, df)
      #add random noise on bl for each visit
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      #generate noisy area
      if(gen.noise){
        d3.fil = g$d3 <= (vis)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis - 0.1/(g$d3[d3.fil]+1)
      }
      #all the pixel value should exceed 0.1
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', vis)
      writeNIfTI(img, file.name)
    }
    df = data.frame(id = str_pad(id, 3, pad = '0'), time = vis+1, ad = 0)
    new.df = rbind(new.df, df)
  }
  return(new.df)
}

#create normal to ad
create.ncad = function(start, end, folder, set = 'train',
                       noise, p, shrink, shrink2){
  
  new.df = data.frame()
  for(id in start:end){
    gen.noise = runif(1) <= p
    s = runif(1, shrink, shrink2)
    fold.name = paste0(cur_path, '/Data/', folder, '/', set, '/', 
                       str_pad(id, 3, pad = '0'))
    dir.create(fold.name)
    bl = array(rnorm(30^3, 2.5, 0.05), c(30, 30, 30))
    for(vis in 1:half.vis){
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = vis, ad = 0)
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      if(gen.noise){
        d3.fil = g$d3 <= (vis)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis 
        - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', vis)
      writeNIfTI(img.cur, file.name)
    }
    for(vis in (half.vis+1):num.vis){
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = vis, ad = 1)
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      d1.fil = g$d1 <= (vis-1)
      d2.fil = g$d2 <= (2*vis-2)
      d1.index = g[d1.fil, c('Var1', 'Var2', 'Var3')]
      d2.index = g[d2.fil, c('Var1', 'Var2', 'Var3')]
      img.cur[as.matrix(d1.index)] = img.cur[as.matrix(d1.index)] - s*0.3*vis - s*0.1/(g$d1[d1.fil]+1)
      img.cur[as.matrix(d2.index)] = img.cur[as.matrix(d2.index)] - s*0.3*vis - s*0.1/(g$d2[d2.fil]+1)
      if(gen.noise){
        d3.fil = g$d3 <= (vis)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', vis)
      writeNIfTI(img.cur, file.name)
    }
    df = data.frame(id = str_pad(id, 3, pad = '0'), time = vis+1, ad = 1)
    new.df = rbind(new.df, df)
  }
  return(new.df)
}

#create ad
create.ad = function(start, end, folder, set = 'train',
                     noise, p = 0.3, shrink = 0.8, shrink2){
  
  new.df = data.frame()
  for(id in start:end){
    gen.noise = runif(1) <= p
    #larger s is associated with more reduction in the pixel 
    #value in c1 and c2 area
    s = runif(1, shrink, shrink2)
    fold.name = paste0(cur_path, '/Data/', folder, '/', set, '/', 
                       str_pad(id, 3, pad = '0'))
    dir.create(fold.name)
    bl = array(rnorm(30^3, 2.5, 0.05), c(30, 30, 30))
    for(vis in 1:num.vis){
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = vis, ad = 1)
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      d1.fil = g$d1 <= (vis)
      d2.fil = g$d2 <= (2*vis)
      d1.index = g[d1.fil, c('Var1', 'Var2', 'Var3')]
      d2.index = g[d2.fil, c('Var1', 'Var2', 'Var3')]
      img.cur[as.matrix(d1.index)] = img.cur[as.matrix(d1.index)] - s*0.3*vis 
                                          - s*0.1/(g$d1[d1.fil]+1)
      img.cur[as.matrix(d2.index)] = img.cur[as.matrix(d2.index)] - s*0.3*vis 
                                          - s*0.1/(g$d2[d2.fil]+1)
      if(gen.noise){
        d3.fil = g$d3 <= (vis)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis 
        - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', vis)
      writeNIfTI(img, file.name)
    }
    df = data.frame(id = str_pad(id, 3, pad = '0'), time = vis+1, ad = 1)
    new.df = rbind(new.df, df)
  }
  return(new.df)
  
}


roi.path = function(folder, set){
  #get the file_names (PIDs)
  cur_path = getwd()
  #should change the folder name according to your need
  folder_name = paste0('/Data/', folder, '/', set)
  image_path = paste0(cur_path, folder_name)
  file_names = dir(image_path)
  image_names = list()
  #for each subject
  for(i in file_names){
    cpath = paste0(image_path, '/', i)
    sub_files = dir(cpath)
    sub_names = c()
    #save all the images
    for(j in 1:length(sub_files)){
      spath = paste0(cpath, '/', sub_files[j])
      sub_names = c(sub_names, spath)
    }
    image_names[[i]] = sub_names
  }
  return(image_names)
}

#save the image information for CNN
save.info = function(img.seq, outcome, folder, set = 'train'){
  
  roi.path = function(folder, set){
    #get the file_names (PIDs)
    cur_path = getwd()
    #should change the folder name according to your need
    folder_name = paste0('/Data/', folder, '/', set)
    image_path = paste0(cur_path, folder_name)
    file_names = dir(image_path)
    image_names = list()
    #for each subject
    for(i in file_names){
      cpath = paste0(image_path, '/', i)
      sub_files = dir(cpath)
      sub_names = c()
      #save all the images
      for(j in 1:length(sub_files)){
        spath = paste0(cpath, '/', sub_files[j])
        sub_names = c(sub_names, spath)
      }
      image_names[[i]] = sub_names
    }
    return(image_names)
  }
  
  image_path = roi.path(folder, set)
  
  
  #list all the subjects by visit
  times = unique(img.seq$time)
  time_id = list()
  for(i in times[-7]){
    time_id[[i]] = img.seq%>%arrange(id, time)%>%group_by(id)%>%
      mutate(index = row_number())%>%ungroup(id)%>%filter(time == i)%>%
      select(id, ad, time, index)
  }
  
  #list all the visit by subject
  id.vis = list()
  for(i in outcome$id){
    id.vis[[i]] = img.seq%>%arrange(id, time)%>%filter(id == i)%>%select(id, ad, time)
  }
  
  
  #save the image by visit
  save.by.visit = function(image_names, folder, set){
    
    for(i in 1:length(times)){
      sub.name = time_id[[i]]
      all.images = c()
      for(j in sub.name$id){
        ind = sub.name%>%filter(id == j)
        all.images = c(all.images, image_names[[j]][ind$index])
      }
      files.data = data.frame(images = all.images, sub.name)
      #should change saving path
      write.csv(files.data, paste('Code/Info/', folder, '/', set, '/Visit/', times[i], '.csv', sep = ''), row.names = FALSE)
    }
    
  }
  
  
  #save the names by rid
  save.by.rid = function(image_names, folder, set){
    
    for(i in names(id.vis)){
      images = image_names[[i]]
      visits = id.vis[[i]]
      #should change saving path
      write.csv(data.frame(images = images, visits), 
                paste('Code/Info/', folder, '/', set, '/RID/', i, '.csv', sep = ''))
    }
    
  }
  save.by.rid(image_path, folder, set)
  save.by.visit(image_path, folder, set)
  write.csv(outcome, paste0('Code/Info/', folder,'/', set, '/outcome.csv'))
}


  
pack.all = function(folder, noise, set = 'train', p, shrink, shrink2 = 1){
    
    nc.df1 = create.nc(1, 300, folder, set, noise, p)
    ncad.df1 = create.ncad(301, 450, folder, set, noise, p, shrink, shrink2)
    ad.df1 = create.ad(451, 600, folder, set, noise, p, shrink, shrink2)
    
    info1 = rbind(nc.df1, ncad.df1, ad.df1)
    info1$id = as.character(info1$id)
    outcome1 = info1%>%filter(time == 5)
    img.seq1 = info1%>%filter(time != 5)
    
    save.info(img.seq1, outcome1, folder, set)

  }
#simulation 1, noise = 0.05, p = 0, shrink = 1
pack.all('Sim1', 0.05, 'train', p = 0, shrink = 1)
pack.all('Sim1', 0.05, 'test', p = 0, shrink = 1)

#simulation 2, noise = 0.1, p = 0, shrink = 1
pack.all('Sim2', 0.1, 'train', p = 0, shrink = 1)
pack.all('Sim2', 0.1, 'test', p = 0, shrink = 1)

#simulation 3, noise = 0.2, p = 0, shrink = 1
pack.all('Sim3', 0.2, 'train', p = 0, shrink = 1)
pack.all('Sim3', 0.2, 'test', p = 0, shrink = 1)

#simulation 4, noise = 0.2, p = 0.5, shrink = 1
pack.all('Sim4', 0.2, 'train', p = 0.5, shrink = 1)
pack.all('Sim4', 0.2, 'test', p = 0.5, shrink = 1)

#simulation 5, noise = 0.2, p = 0.5, shrink = 0.5-0.7
pack.all('Sim5', 0.2, 'train', p = 0.5, shrink = 0.5, shrink2 = 0.7)
pack.all('Sim5', 0.2, 'test', p = 0.5, shrink = 0.5, shrink2 = 0.7)

#simulation 6, noise = 0.2, p = 0.5, shrink = 0.2-0.4
pack.all('Sim6', 0.2, 'train', p = 0.5, shrink = 0.2, shrink2 = 0.4)
pack.all('Sim6', 0.2, 'test', p = 0.5, shrink = 0.2, shrink2 = 0.4)

#simulation 7, noise = 0.2, p = 0.7, shrink = 0.1-0.2
pack.all('Sim7', 0.2, 'train', p = 0.7, shrink = 0.1, shrink2 = 0.2)
pack.all('Sim7', 0.2, 'test', p = 0.7, shrink = 0.1, shrink2 = 0.2)

#simulation 8, noise = 0.2, p = 0.7, shrink = 0 (test model)
pack.all('Sim8', 0.2, 'train', p = 0.7, shrink = 0, shrink2 = 0)
pack.all('Sim8', 0.2, 'test', p = 0.7, shrink = 0, shrink2 = 0)

#simulation 9, noise = 0.2, p = 0.7, shrink = 0.1
pack.all('Sim9', 0.2, 'train', p = 0.7, shrink = 0.1, shrink2 = 0.1)
pack.all('Sim9', 0.2, 'test', p = 0.7, shrink = 0.1, shrink2 = 0.1)

#simulation 10, noise = 0.2, p = 0.7, shrink = 0.05
pack.all('Sim10', 0.2, 'train', p = 0.7, shrink = 0.05, shrink2 = 0.05)
pack.all('Sim10', 0.2, 'test', p = 0.7, shrink = 0.05, shrink2 = 0.05)

#simulation 11, noise = 0.2, p = 0.7, shrink = 0.1-1
pack.all('Sim11', 0.2, 'train', p = 0.7, shrink = 0.1, shrink2 = 1)
pack.all('Sim11', 0.2, 'test', p = 0.7, shrink = 0.1, shrink2 = 1)

#simulation 12, noise = 0.2, p = 0.7, shrink = 0.02
pack.all('Sim12', 0.2, 'train', p = 0.7, shrink = 0.02, shrink2 = 0.02)
pack.all('Sim12', 0.2, 'test', p = 0.7, shrink = 0.02, shrink2 = 0.02)

#simulation 13, noise = 0.2, p = 0.7, shrink = 0.02
pack.all('Sim13', 0.2, 'train', p = 0.7, shrink = 0.03, shrink2 = 0.03)
pack.all('Sim13', 0.2, 'test', p = 0.7, shrink = 0.03, shrink2 = 0.03)

#simulation 14, noise = 0.2, p = 0.7, shrink = 0.01
pack.all('Sim14', 0.2, 'train', p = 0.7, shrink = 0.01, shrink2 = 0.01)
pack.all('Sim14', 0.2, 'test', p = 0.7, shrink = 0.01, shrink2 = 0.01)


#function for building graph information
build.graph = function(sim.run = 'Sim3', build = 'train'){
  
  cur.path = getwd()
  folder = paste0(cur.path, '/Code/Info/', sim.run, '/', build)
  outcome = read.csv(paste0(folder, '/outcome.csv'))
  rids = dir(paste0(folder, '/RID'))
  
  #initiate dataframe for edges and graphs
  all.edges = data.frame()
  all.graphs = data.frame()
  for (rid in rids){
    file = read.csv(paste0(folder, '/RID/', rid))
    #create edge information
    ##combination of row number
    num_nodes = nrow(file)
    src.tar = combn(1:num_nodes, 2)
    times = file$time
    ##weights equal to difference of time
    weights = 1/(times[src.tar[2,]] - times[src.tar[1,]])
    graph_id = substr(rid, 1, 3)
    edges = data.frame(graph_id = graph_id, src = src.tar[1,],
                      dst = src.tar[2,], weight = weights)
    
    #create graph information
    item = filter(outcome, id == as.numeric(graph_id))
    graphs = data.frame(graph_id = graph_id, label = item$ad, num_nodes = num_nodes)
    
    #combine information
    all.graphs = rbind(all.graphs, graphs)
    all.edges = rbind(all.edges, edges)
  }
    
    save.folder = paste0(cur.path, '/Code/Info/SimGraph/', sim.run, '/', build)
    write.csv(all.graphs, paste0(save.folder, '/graphs.csv'), quote = FALSE)
    write.csv(all.edges, paste0(save.folder, '/edges.csv'), quote = FALSE)
  
}

#create graph information for Sim3

build.graph('Sim3', 'train')
build.graph('Sim3', 'test')

#create fake data for graph
nodes.feature = function(start = 1, id){
  
  nums = 4
  features = c()
  bl = rnorm(128, mean = 5, sd = 0.1)
  if(start > 1){
    for(i in 1:(start-1)){
      new = bl + rnorm(128, 0, 0.05)
      features = rbind(features, new)
    }
  }
  if(start < 5){
    for(i in start:nums){
      new = bl - rnorm(128, 0.8*i, 0.05)
      features = rbind(features, new)
    }
  }
  features = data.frame(features)
  
  features$graph_id = id
  rownames(features) = NULL
  return(features)

}
#create normal, start = 5
nodes.set = function(starts = c(5,3,1)){
  df = data.frame()
  for(i in 1:300){
    id = str_pad(i, 3, pad = '0')
    f1 = nodes.feature(start = starts[1], id)
    df = rbind(df, f1)
  }
  for(i in 301:450){
    id = str_pad(i, 3, pad = '0')
    f2 = nodes.feature(start = starts[2], id)
    df = rbind(df, f2)
  }
  for(i in 451:600){
    id = str_pad(i, 3, pad = '0')
    f3 = nodes.feature(start = starts[3], id)
    df = rbind(df, f3)
  }
  return(df)
  
}
train_nodes = nodes.set(c(5,5,5))
test_nodes = nodes.set(c(5,5,5))
write.csv(train_nodes, 'Code/Info/SimGraph/Sim3/train/nodes.csv', quote = FALSE)
write.csv(test_nodes, 'Code/Info/SimGraph/Sim3/test/nodes.csv', quote = FALSE)
#making plots
for(k in 1:4){
  img_p = paste0('Y:/PhD/Fall 2020/RA/Data/Sim10/test/580/',k,'.nii')
  tmp = readNIfTI(img_p)
  assign(paste('img',k, sep = ''), tmp)
}


par(mfrow = c(2,2))
image(img1[,,20], col = gray.colors(30), axes = FALSE, main = 'Visit 1')
image(img2[,,20], col = gray.colors(30), axes = FALSE, main = 'Visit 2')
image(img3[,,20], col = gray.colors(30), axes = FALSE, main = 'Visit 3')
image(img4[,,20], col = gray.colors(30), axes = FALSE)
mtext("Area B", side = 3, line = -14, outer = TRUE)

noise1 = readNIfTI('Y:/PhD/Fall 2020/RA/Data/graphSim6/train/245/1.nii')
noise2 = readNIfTI('Y:/PhD/Fall 2020/RA/Data/graphSim6/train/245/2.nii')
noise3 = readNIfTI('Y:/PhD/Fall 2020/RA/Data/graphSim6/train/245/3.nii')
noise4 = readNIfTI('Y:/PhD/Fall 2020/RA/Data/graphSim6/train/245/4.nii')
noise5 = readNIfTI('Y:/PhD/Fall 2020/RA/Data/graphSim6/train/245/5.nii')

tempfdg = readNIfTI('temppet.nii')
image(noise1[,,10], col = gray.colors(30), axes = FALSE)
image(noise2[,,10], col = gray.colors(30), axes = FALSE)
image(noise3[,,10], col = gray.colors(30), axes = FALSE)
image(noise4[,,10], col = gray.colors(30), axes = FALSE)
image(noise5[,,10], col = gray.colors(30), axes = FALSE)

