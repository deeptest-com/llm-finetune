import codecs

from src.lib.file import traverse_files

encode_in = 'GB18030'
encode_out = 'utf-8'

src = traverse_files("data/aerospace", num=10)

for filename in src:
    with codecs.open(filename=filename, mode='r', encoding=encode_in) as fi:
        data = fi.read()

        filename_new = filename.replace("aerospace", "aerospace_utf8")

        with open(filename_new, mode='w', encoding=encode_out) as fo:
            fo.write(data)
            fo.close()
