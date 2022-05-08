Requirements:
```
sudo apt update
sudo apt install nodejs npm
sudo npm install --global embedme
```

Render documentation
```
npx embedme --strip-embed-comment --stdout docs/documentation.md > docs/.tmp.md && \
pandoc --standalone --toc --pdf-engine=xelatex \
--resource-path=docs \
-H docs/head.tex \
--wrap auto --highlight-style espresso \
docs/.tmp.md \
-o pdf-documentation.pdf;\
rm docs/.tmp.md
```

Render technical documentation
```
pandoc --standalone --toc --pdf-engine=xelatex \
--resource-path=. \
-H docs/head.tex \
--wrap auto --highlight-style espresso \
README.md \
-o docs/pdf-readme-documentation.pdf
pdfunite docs/pdf-readme-documentation.pdf docs/pdf-api-docs.pdf pdf-technical-documentation.pdf
```

```
```

To resize images to 2000px:
```
mogrify -resize 2000x *.jpg *.png *.jpeg
```