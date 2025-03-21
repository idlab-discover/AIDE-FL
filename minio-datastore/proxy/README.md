# Minio proxy

Deze applicatie is de middleman tussen de Minio objectstore en de FL client met als taak om datasets te downloaden van Kaggle moesten die nog niet aanwezig zijn in de objectstore. Indien wel aanwezig, wordt de dataset geforward.