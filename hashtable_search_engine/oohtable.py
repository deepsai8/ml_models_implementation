"""
A HashTable represented as a list of lists with open hashing.
Each bucket is a list of (key,value) tuples
"""

class HashTable:
    def __init__(self, nbuckets):
        """Init with a list of nbuckets lists"""
        self.nbuckets = nbuckets
        self.buckets = [[] for i in range (nbuckets)]
        self.size = 0


    def __len__(self):
        """
        number of keys in the hashable
        """
        return self.size


    def __setitem__(self, key, value):
        """
        Perform the equivalent of table[key] = value
        Find the appropriate bucket indicated by key and then append (key,value)
        to that bucket if the (key,value) pair doesn't exist yet in that bucket.
        If the bucket for key already has a (key,value) pair with that key,
        then replace the tuple with the new (key,value).
        Make sure that you are only adding (key,value) associations to the buckets.
        The type(value) can be anything. Could be a set, list, number, string, anything!
        """
 
        h = hash(key) % len(self.buckets)
        bucket = self.buckets[h]
        keyOfBucket = self.bucket_indexof(key)
        print(f'keyOfBucket: {keyOfBucket}')
        if keyOfBucket == None:
            bucket.append((key, value))
            self.size += 1
        else:
            bucket[keyOfBucket]=(key,value)


    def __getitem__(self, key):
        """
        Return the equivalent of table[key].
        Find the appropriate bucket indicated by the key and look for the
        association with the key. Return the value (not the key and not
        the association!). Return None if key not found.
        """

        h = hash(key) % len(self.buckets)
        bucket = self.buckets[h]
        for k, v in bucket:
            if key == k:
                return v
        else:
            return None

    
    def __contains__(self, key):
        """
        check if key exist in hash table
        """
        h = hash(key) % len(self.buckets)
        for item in self.buckets[h]:
            if item[0] == key:
                return True
        return False


    def __iter__(self):
        '''
        To iterate over keys
        '''
        # for k, v in self.items():
        #     yield k
        return iter(self.keys())


    def keys(self):
        """
        return all keys in the hashtable

        Returns
        -------
        elems : TYPE
            DESCRIPTION.
        """
        # return self.__iter__()
        op = []
        for k, v in self.items():
            op.append(k)
        return op


    def items(self):
        """
        returns all values in the hashable
        """
        op = []
        for b in self.buckets:
            if not b:
                continue
            for i in b:
                op.append(i)
        return op


    def __repr__(self):
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
        str1=''
        for i in range(len(self.buckets)):
            if len(self.buckets[i])==0:
            #print(i)
                str1=str1+format(i,'0>4d')+'->'+"\n"
            else:
                str1=str1+format(i,'0>4d')+'->'
                for j,d in enumerate(self.buckets[i]):
                    if j<(len(self.buckets[i])-1):
                        str1=str1+f"{d[0]}:{d[1]}"+', '
                    #print(str1)
                    else:
                        str1=str1+f"{d[0]}:{d[1]}" + "\n"
        return str1


    def __str__(self):
        """
        Return what str(table) would return for a regular Python dict
        such as {parrt:99}. The order should be in bucket order and then
        insertion order within each bucket. The insertion order is
        guaranteed when you append to the buckets in htable_put().
        """
        str1=''
        list2=[]
        for i in range(len(self.buckets)):
            for j, d in enumerate(self.buckets[i]):
                list2.append(f"{d[0]}:{d[1]}")

        return '{'+", ".join(list2)+'}'


    def bucket_indexof(self, key):
        """
        You don't have to implement this, but I found it to be a handy function.

        Return the index of the element within a specific bucket; the bucket is:
        table[hashcode(key) % len(table)]. You have to linearly
        search the bucket to find the tuple containing key.
        """
        h = hash(key) % len(self.buckets)
        bucket = self.buckets[h]
        for i in range(len(bucket)):
            if bucket[i][0]==key:
                return i
        return None
