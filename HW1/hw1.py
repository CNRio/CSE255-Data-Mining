# Name: Yue Wang
# Email: yuw383@eng.ucsd.edu
# PID: A53090624
from pyspark import SparkContext
sc = SparkContext()
# Your program here

from string import punctuation
def printOutput(n,freq_ngramRDD):
    top=freq_ngramRDD.take(5)
    print '\n============ %d most frequent %d-grams'%(5,n)
    print '\nindex\tcount\tngram'
    for i in range(5):
        print '%d.\t%d: \t"%s"'%(i+1,top[i][0],' '.join(top[i][1]))

def rmPunc(s):
    t = ''
    for c in s:
        if c not in punctuation:
            t += c
        else:
            t += ' '
    return t

textRDD = sc.newAPIHadoopFile('/data/Moby-Dick.txt',
                              'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
                              'org.apache.hadoop.io.LongWritable',
                              'org.apache.hadoop.io.Text',
                               conf={'textinputformat.record.delimiter': "\r\n\r\n"}) \
            .map(lambda x: x[1])

sentences=textRDD.map(lambda x: x.replace('\r\n', ' ')) \
                 .map(rmPunc).map(lambda word: word.lower()) \
                 .flatMap(lambda x: x.split(". "))

for n in range(1,6):
    if n==1:
        ngramRDD = sentences.map(lambda x:x.split()) \
            .flatMap(lambda x: [(tuple([x[i]]),1) for i in range(0,len(x))]) 
    elif n==2:
        ngramRDD = sentences.map(lambda x:x.split()) \
            .flatMap(lambda x: [((x[i],x[i+1]),1) for i in range(0,len(x)-1)])
    elif n==3:
        ngramRDD = sentences.map(lambda x:x.split()) \
            .flatMap(lambda x: [((x[i],x[i+1],x[i+2]),1) for i in range(0,len(x)-2)])
    elif n==4:
        ngramRDD = sentences.map(lambda x:x.split()) \
            .flatMap(lambda x: [((x[i],x[i+1],x[i+2],x[i+3]),1) for i in range(0,len(x)-3)])
    else:
        ngramRDD = sentences.map(lambda x:x.split()) \
            .flatMap(lambda x: [((x[i],x[i+1],x[i+2],x[i+3],x[i+4]),1) for i in range(0,len(x)-4)])
        
    freq_ngramRDD = ngramRDD.reduceByKey(lambda a,b: a+b) \
            .map(lambda (c,v): (v,c)) \
            .sortByKey(False)    
    printOutput(n,freq_ngramRDD)

