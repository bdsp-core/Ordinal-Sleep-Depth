def find_consecutive_numbers(nums):
    """
    returns nx2 array with starting and ending points of consecutive numbers
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    consecutive = np.array(list(zip(edges, edges))) 
    if consecutive != []:
        consecutive[:,1] +=1
    return consecutive

def find_events(y_true_samplewise,positive_label_array):
    """
    pre-allocate array
    """
    events = np.array([0,0])
    events_labels = 0
    #loop over positive classes and add to arrays
    for lab in positive_label_array:
        #find positive label
        events_temp = find_consecutive_numbers(np.where(y_true_samplewise==lab)[0])
        #add to array
        if events_temp.shape[0]!=0:
            events = np.vstack((events,events_temp))
            events_labels = np.append(events_labels,np.ones(len(events_temp))*lab)