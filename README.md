# MVSEC dataset

## How to build the website

	git checkout website

then edit the files in the "content" directory. When done:

	cd website
	hugo server -D

To test, point your browser to localhost:1313/mvsec

If you are happy with the results, first commit the source code to
the "website" branch:

	git commit -a
	git push

Then commit/push the compiled web pages to the project page:

	cd public
	git commit -a
	git push




