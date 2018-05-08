from __future__ import print_function

import time


def monitor_terminated_processes(workers, activeEpisodes, episodeQueue):
    """
        Watch list of workers (spawn processes) and if worker is not alive then 
        check if it had terminated abnormally. In this case, uncompleted episode of work 
        is moved to Queue for further processing.
        @param workers: list of workers (spawn processes)
        @param activeEpisodes: dictionary with key=process.id and value=episodeNumber
        @param episodeQueue: queue of episodes of work to be processed
    """
    restarted = {} # dictionary with key=episodeNumber and value=times the episode was restarted
    # episode will not be restarted more than 3 times
    while True:
        time.sleep(5)
        alive = 0
        for proc in workers:
            if proc.is_alive():
                alive += 1
            else:
                if proc.pid in activeEpisodes: # abnormal termination, work is not done
                    print("[", proc.pid, "] process terminated abnormally")
                    episodeNum = activeEpisodes[proc.pid]
                    if episodeNum in restarted and restarted[episodeNum]>=3:
                        print("[", proc.pid, "] episode", episodeNum, " was restarted too many times, skip it")
                    else:
                        print("[", proc.pid, "] return episode", episodeNum, "to the Queue")
                        episodeQueue.put(episodeNum)
                        if episodeNum not in restarted:
                            restarted[episodeNum] = 1
                        else:
                            restarted[episodeNum] + 1
                    # delete lock record
                    activeEpisodes.pop(proc.pid, None)
        if alive==0:
            break
