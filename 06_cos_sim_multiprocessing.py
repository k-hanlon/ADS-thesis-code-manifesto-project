# # Main ideas

# # worker function: gets access to numpy array of all embeddings, and gets current row from queue
# # calculates cosine similarities, creates a sparse vector with the ones over threshold
# # returns this vector and its row index


# # main function
# # embeddings are shared between workers using something like this:
#         # Create a NumPy array of type float32
#         data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
#         # Create a shared memory array from the NumPy array
#         shared_array_base = mp.Array('f', data)
#         # Convert the shared memory array to a NumPy array
#         shared_array = np.frombuffer(shared_array_base.get_obj(), dtype=np.float32)

# # create a queue with all row numbers and some Nones

# # start worker process, save their results in a queue (so this will be a queue of sparse vectors, hopefully not a memory problem...)
# # sort it
#     results = [result_queue.get() for _ in range(num_rows)]
#     # Sort the results based on the original row indices
#     results.sort(key=lambda x: x[0])
#     # build a full sparse matrix from it (maybe like this, but I think GPT used dicitonaries. And 200K dicts might be annoying)
#     # Populate the list of row data using the sorted results
#     for row_index, row_data in results:
#         row = csr_matrix(([row_data[j] for j in range(num_rows)], ([0] * len(row_data), list(row_data.keys()))), shape=(1, num_rows))
#         rows.append(row)

#     # Concatenate the rows to form the complete sparse matrix
#     full_matrix = vstack(rows)




import numpy as np
import time
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix, vstack
import multiprocessing as mp
import argparse

def calc_cos_sim(emb, queue, shared_list):
    n = len(emb)
    while(True):
        i = queue.get() # this is the index of the row for the similarity matrix we are focusing on now
        #print(i)
        if i is None:
            # we reached the end of the queue, end this worker process
            break

        # we will save the relevant similarity values in a sparse vector
        cosine_sim_vector = csr_matrix((1, n), dtype=np.float32)

        # Compute only the upper triangle
        for j in range(i + 1, n):
            # Cosine similarity calculation
            cos_sim = np.dot(emb[i], emb[j])
            if cos_sim > 0.8 or cos_sim < -0.8:
                cosine_sim_vector[0, j] = cos_sim

        # add the sparse vector together with its index, so that we can sort it later correctly
        shared_list.append((i, cosine_sim_vector))
        # if i > 100:
        #     break




def main_process(args):
    # load the sentence embeddings:
    embeddings = np.load("/bayes_data/MS-2023/hanlon/data/mp_full_sts_embeddings.npy")
    #embeddings = embeddings[0:50]
    # ideally this array is shared by the processes, but for now we will just pass copies to all the workers...

    # Create a query with all the indecies of the embeddings:
    q = mp.Queue()
    for j in range (0, len(embeddings)):
        q.put(j)
    # add None elements, so the workers know they are done:
    for i in range(0, args.workers):
        q.put(None)

    # create a shared list for the results
    manager = mp.Manager()
    shared_list = manager.list()
    # start the worker processes:
    workers = []
    for i in range(args.workers):        
        worker = mp.Process(target=calc_cos_sim, args=(embeddings, q, shared_list))
        worker.start()
        workers.append(worker)

    # let workers finish:
    for worker in workers:
        worker.join()

    # now shared_list contains the sparse vectors, but order might be wrong
    sorted_vectors = [x for _, x in sorted(shared_list, key=lambda tup: tup[0])]
    combined_sparse_matrix = vstack(sorted_vectors)
    print("How many values are stored:", combined_sparse_matrix.count_nonzero())
    save_npz("/bayes_data/MS-2023/hanlon/data/cos_sim_output.npz", combined_sparse_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate cosine similarity between sentence embeddings',
        epilog = 'Example: bash /opt/local/bin/run_job.sh --partition "cpu-shannon" --cpus-per-task 1 --script cos_sim_mp.py -- -w 1'
    )
    parser.add_argument('--workers', '-w',
                        default='1',
                        type = int,
                        help='Number of parallel processes to use') 
    args = parser.parse_args()

    start_time = time.time()
    main_process(args)
    end_time = time.time()
    final_time = end_time - start_time
    print("Total running time: %3.2f [s]" % (final_time) )
    #print("For all in h:", (final_time * 202187)/60/60 /args.workers /2 /100)
    print("With", args.workers, "workers.")