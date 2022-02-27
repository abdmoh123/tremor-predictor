def main():
    in_file = open("N_pointingxyzAI_mag1_force01_an02.txt", "r")
    out_file = open("N_pointingxyzAI_mag1_force01_an02.csv", "a")
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


if __name__ == '__main__':
    main()
