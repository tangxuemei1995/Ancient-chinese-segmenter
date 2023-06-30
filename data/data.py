from langconv import *
def fan_jian(char):
    '''
    繁简转化
    '''

    jian_char = Converter('zh-hans').convert(char)

    return jian_char
    
f = open('./test.txt','w',encoding='utf-8')

for line in open('./zztj/1.txt'):
        new_line  = 'SGW\tO' +'\n'
        count =0
        # print(line)
        if len(line) > 150:

            new_line +=  '。' + '\t' + 'O'+'\n'
            count +=1
            new_line += '/SGW\tO' + '\n\n'
            f.write(new_line)
            new_line = 'SGW\tO' + '\n'

            print(len(line))
            lines = line.strip().split('。')
            for ju in lines:
                if ju != '':
                    ju += '。'
                    for x in ju.strip():
                        x = fan_jian(x)
                        new_line += x + '\t' + 'O' + '\n'
                    new_line += '/SGW\tO' + '\n\n'
                    f.write(new_line)
                    new_line = 'SGW\tO' + '\n'
            new_line += '。' + '\t' + 'O' + '\n'
            new_line += '/SGW\tO' + '\n\n'
            f.write(new_line)
            new_line = 'SGW\tO' + '\n'

        else:
            for x in line.strip():
                x = fan_jian(x)
                new_line += x + '\t' + 'O'+'\n'
            
            new_line += '/SGW\tO'+'\n\n'
            f.write(new_line)
            new_line  = 'SGW\tO' +'\n'
