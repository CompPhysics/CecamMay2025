#!/bin/sh
set -x

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

if [ $# -eq 0 ]; then
echo 'bash make.sh slides1|slides2'
exit 1
fi

name=$1
rm -f *.tar.gz

opt="--encoding=utf-8"
# Note: Makefile examples contain constructions like ${PROG} which
# looks like Mako constructions, but they are not. Use --no_mako
# to turn off Mako processing.
opt="--no_mako"

rm -f *.aux


html=${name}-reveal
system doconce format html $name --pygments_html_style=perldoc --keep_pygments_html_bg --html_links_in_new_window --html_output=$html $opt
system doconce slides_html $html reveal --html_slide_theme=beige

# Plain HTML documents

html=${name}-solarized
system doconce format html $name --pygments_html_style=perldoc --html_style=solarized3 --html_links_in_new_window --html_output=$html $opt
system doconce split_html $html.html --method=space10

html=${name}
system doconce format html $name --pygments_html_style=default --html_style=bloodish --html_links_in_new_window --html_output=$html $opt
system doconce split_html $html.html --method=space10

# Bootstrap style
html=${name}-bs
system doconce format html $name --html_style=bootstrap --pygments_html_style=default --html_admon=bootstrap_panel --html_output=$html $opt
system doconce split_html $html.html --method=split --pagination --nav_button=bottom

# IPython notebook
system doconce format ipynb $name $opt

# LaTeX Beamer slides
beamertheme=red_plain
system doconce format pdflatex $name --latex_title_layout=beamer --latex_table_format=footnotesize $opt
system doconce ptex2tex $name envir=minted
# Add special packages
doconce subst "% Add user's preamble" "\g<1>\n\\usepackage{simplewick}" $name.tex
system doconce slides_beamer $name --beamer_slide_theme=$beamertheme
system pdflatex -shell-escape ${name}
system pdflatex -shell-escape ${name}
cp $name.pdf ${name}-beamer.pdf
cp $name.tex ${name}-beamer.tex



# Publish
dest=../pub
if [ ! -d $dest/$name ]; then
mkdir $dest/$name
mkdir $dest/$name/pdf
mkdir $dest/$name/html
mkdir $dest/$name/ipynb
fi
cp ${name}*.pdf $dest/$name/pdf
cp -r ${name}*.html ._${name}*.html reveal.js $dest/$name/html

# Figures: cannot just copy link, need to physically copy the files
if [ -d fig-${name} ]; then
if [ ! -d $dest/$name/html/fig-$name ]; then
mkdir $dest/$name/html/fig-$name
fi
cp -r fig-${name}/* $dest/$name/html/fig-$name
fi

cp ${name}.ipynb $dest/$name/ipynb
ipynb_tarfile=ipynb-${name}-src.tar.gz
if [ ! -f ${ipynb_tarfile} ]; then
cat > README.txt <<EOF
EOF
This IPython notebook ${name}.ipynb does not require any additional
programs.
EOF
tar czf ${ipynb_tarfile} README.txt
fi
cp ${ipynb_tarfile} $dest/$name/ipynb






