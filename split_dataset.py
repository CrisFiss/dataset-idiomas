with open('german.txt') as txt:
    texto = txt.read()

sentences= texto.split('$$$')
sentences.remove('')
print(sentences)

sorting = True
outer_count = 1
line_count = 0
while sorting:
    count = 0
    increment = (outer_count-1) * 1
    left = len(sentences) - increment
    file_name = "ep-00-01-" + str(outer_count * 1) + ".txt"
    hold_new_lines = []
    if left < 1:
        while count < left:
            hold_new_lines.append(sentences[line_count])
            count += 1
            line_count += 1
        sorting = False
    else:
        while count < 1:
            hold_new_lines.append(sentences[line_count])
            count += 1
            line_count += 1
    outer_count += 1
    with open(file_name,'w') as next_file:
        for row in hold_new_lines:
            next_file.write(row)