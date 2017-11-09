

#########################################################################################################
### Deprecated, please don't use!!
################################################x########################################################
    
def reader(queue, file, epochs):
    for i in range(epochs):
        with open(file) as f:
            while True:
                try:
                    sentence = []
                    while True:
                        line = f.readline()[:-1].split('\t')
                        if line[0] != "":
                            sentence.append(line)
                        else:
                            break
                    queue.put(sentence)
                except e:
                    print(e)
                    
                    
def vocabulary(path):
    with open(path) as f:
        return sorted(set([char for char in f.readline().replace("\n","")]))

                    
#tensorflow data helpers
def char_split(input_line, delimiter=''):
    source, target = tf.string_split(input_line, delimiter=delimiter)
    return source, target

def decode(string):
    string=str(string)
    return string.decode('utf-8')

def read_line(filename_queue):
    global SPLIT_CHAR
    reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""]]
    source, target = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=SPLIT_CHAR)
    
    return {"in": source,"out": target}
