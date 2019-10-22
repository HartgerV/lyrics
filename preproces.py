import csv
import itertools
from pandas import (
    DataFrame, HDFStore
)
from annoy import AnnoyIndex
from document_embedding import document_embedding

song_store = HDFStore('songs.h5')
songs = []
embedding_size = 300
t = AnnoyIndex(embedding_size, 'angular')

with open('lyrics.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    index = 0
    for row in itertools.islice(csv_reader, 0, 380000, 10):
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            fields = row
        else:
            lyrics = row[5]
            if line_count == 1:
                testPhrase = lyrics
            if len(lyrics) > 0:
                embedding = document_embedding(lyrics)
                t.add_item(index, embedding)
                songs.append((row[1],row[3],row[5]))
                index += 1
            line_count += 1

    print(f'Processed {line_count} lines.')

t.build(10) # 10 trees
t.save('test.ann')
song_df = DataFrame(songs, columns = ['songtitle' , 'artist', 'lyrics'])
song_store.put('/mat', song_df)
