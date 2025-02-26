from clearml import Task
import torch
import torch.distributed as dist

def run(rank, size):
    print('World size is ', size)
    tensor = torch.zeros(1)
    if rank == 0:
        for i in range(1, size):
            tensor += 1
            dist.send(tensor=tensor, dst=i)
            print('Sending from rank ', rank, ' to rank ', i, ' data: ', tensor[0])
    else:
        dist.recv(tensor=tensor, src=0)
        print('Rank ', rank, ' received data: ', tensor[0])

if __name__ == '__main__':
    task = Task.init(project_name='Multi-node testing', task_name='multinode_test')
    task.execute_remotely(queue_name='default')
    config = task.launch_multi_node(2)
    print(config)
    dist.init_process_group('nccl')
    run(config.get('node_rank'), config.get('total_num_nodes'))