def initializeOutputFile(dataset):
    output = open('output/' + dataset+'.csv', 'w')
    output.write('nframes,nfeats,k,k_relu,RNN Layer,epoch_size,monitor,accuracy,stdev\n')

def write(dataset,content):
    output = open('output/' + dataset+'.csv', 'a')
    output.write(content)

def errorWrite(content):
    error = open('logs/error.txt', 'a')
    error.write(content + '\n')
