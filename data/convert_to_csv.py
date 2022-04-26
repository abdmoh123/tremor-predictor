import os


def main():
    folder_name = "./Surgeon Tracing/"
    directory_name = "C:/Users/Abdul/OneDrive - Newcastle University/Stage 3/Obsidian Vault/EEE3095-7 Individual Project and Dissertation/Tremor ML/data/" + folder_name[2:]
    directory = os.fsencode(directory_name)
    file = ""

    filenames = []
    # if file is specified, program will look for the 1 file instead of a directory
    if len(file) > 0:
        filenames.append(file)
    else:
        # puts all txt files' names in a list
        for file in os.listdir(directory):
            filenames.append(os.fsdecode(file))

    # converts each txt file to a list
    for filename in filenames:
        filename = filename[:(len(filename) - 4)]
        print("Converted file:", filename)

        in_file = open(folder_name + filename + ".txt", "r")
        out_file = open(folder_name + filename + ".csv", "a")
        for line in in_file:
            content_array = line.split()
            content = ""
            for x in range(len(content_array)):
                content += content_array[x]
                if x < (len(content_array) - 1):
                    content += ","
                else:
                    content += "\n"
            out_file.write(content)
        out_file.close()
        in_file.close()

    # removes the 1 file if specified instead of a directory
    if len(file) > 0:
        os.remove(file)
    else:
        # removes all text files in the directory
        for filename in filenames:
            os.remove(folder_name + filename)


if __name__ == '__main__':
    main()
