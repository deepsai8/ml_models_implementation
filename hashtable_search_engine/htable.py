"""
A hashtable represented as a list of lists with open hashing.
Each bucket is a list of (key,value) tuples
"""

def htable(nbuckets):
    """Return a list of nbuckets lists"""
    return [[] for i in range(nbuckets)]


def hashcode(o):
    """
    Return a hashcode for strings and integers; all others return None
    For integers, just return the integer value.
    For strings, perform operation h = h*31 + ord(c) for all characters in the string
    """
    if type(o) == type(''):
        h = 0
        for c in o: 
            h = h*31 + ord(c)
        return h
    elif type(o) == type(0):
        return o
    else:
        return None


def bucket_indexof(table, key):
    """
    You don't have to implement this, but I found it to be a handy function.
    Return the index of the element within a specific bucket; the bucket is:
    table[hashcode(key) % len(table)]. You have to linearly
    search the bucket to find the tuple containing key.
    """
    


def htable_put(table, key, value):
    """
    Perform the equivalent of table[key] = value
    Find the appropriate bucket indicated by key and then append (key,value)
    to that bucket if the (key,value) pair doesn't exist yet in that bucket.
    If the bucket for key already has a (key,value) pair with that key,
    then replace the tuple with the new (key,value).
    Make sure that you are only adding (key,value) associations to the buckets.
    The type(value) can be anything. Could be a set, list, number, string, anything!
    """
    bucket_index = hashcode(key) % len(table)
    bucket = table[bucket_index]
    #print(f'bucket_index: {bucket_index}, bucket: {bucket}')
    
    if len(bucket) == 0:
        #print(f'len(bucket) = {len(bucket)}, new value added')
        bucket.append((key,value))
    else:
        needed = len(bucket)
        for i in range(len(bucket)):
            #print(f'len(bucket) = {len(bucket)}, i: {i}, bucket[i]: {bucket[i]}')
            if key == bucket[i][0]:
                #print(f'key: {key} = bucket[i][0]: {bucket[i][0]}, therefore value updated')
                bucket[i] = (key, value)
                break
            else:
                needed -= 1
                continue
        if needed == 0:
            #print(f'key: {key} not= bucket[i][0]: {bucket[i][0]}, so value appended')
            bucket.append((key,value))
            

def htable_get(table, key):
    """
    Return the equivalent of table[key].
    Find the appropriate bucket indicated by the key and look for the
    association with the key. Return the value (not the key and not
    the association!). Return None if key not found.
    """
    bucket_index = hashcode(key) % len(table)
    bucket = table[bucket_index]
    
    flag = 1
    for i in range(len(bucket)):
        if key == bucket[i][0]:
            flag = 0
            return bucket[i][1]
        else:
            continue
    return None


def htable_buckets_str(table):
    """
    Return a string representing the various buckets of this table.
    The output looks like:
        0000->
        0001->
        0002->
        0003->parrt:99
        0004->
    where parrt:99 indicates an association of (parrt,99) in bucket 3.
    """
    output =''
    
    for i in range(len(table)):
        buc = table[i]
        c = len(str(i))
        cind = ''
        for n in range(4):
            if c < (4 - n):
                cind += '0'
            else:
                cind += str(i)[c - 4 + n]
        
        output += cind
        output += f'->' + ''
        for j in table[i]:
            output += f'{j[0]}:{j[1]}'
            if table[i].index(j) < len(table[i]) -1:
                output += ', '

        output += '\n'
    return output



def htable_str(table):
    """
    Return what str(table) would return for a regular Python dict
    such as {parrt:99}. The order should be in bucket order and then
    insertion order within each bucket. The insertion order is
    guaranteed when you append to the buckets in htable_put().
    """
    output =''
    output += '{'
    for i in range(len(table)):
        buc = table[i]
        c = len(str(i))

        for j in table[i]:
            output += f'{j[0]}:{j[1]}, '
    output = output.strip(', ')
    output += '}'
    return output