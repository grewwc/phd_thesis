package main

import (
	"containerW"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/net/html"
)

const (
	rootHtml  = "http://archive.stsci.edu/pub/kepler/lightcurves"
	rootStore = "C:/Users/User/dev/data"
)

var max = make(chan struct{}, 8)
var subRoots []string
var mu sync.Mutex
var wg sync.WaitGroup

var failed = containerW.NewDeque()

var debugCount int32 = 1

func searchDownload(curHtml string) {
	fmt.Printf("visiting %q\n", curHtml)

	resp, err := http.Get(curHtml)
	if err != nil {
		log.Printf("url: %q, err: %s\n", curHtml, err)
		failed.PushBack(curHtml)
		return
	}
	defer resp.Body.Close()
	node, err := html.Parse(resp.Body)
	if err != nil || node == nil {
		log.Printf("parsing %q error \n", curHtml)
		return
	}

	var found []string
	isEnd := processHtml(node, &found)

	if len(found) == 0 {
		return
	}

	if isEnd {
		for i := range found {
			found[i] = curHtml + "/" + found[i]

			// subRoots = append(subRoots, found...)
			wg.Add(1)
			go func(url string) {
				defer wg.Done()
				defer func() {
					<-max
				}()
				max <- struct{}{}
				download(url)
			}(found[i])
		}
		return
	}

	// not the end point of html file
	for _, val := range found {
		searchDownload(curHtml + "/" + val)
	}
}

func download(url string) {
	fmt.Printf("downloading %q\n", url)
	defer func() {
		fmt.Println("finised")
	}()
	filename := strings.TrimPrefix(url, rootHtml)
	filename = filepath.Join(rootStore, filename)
	resp, err := http.Get(url)
	if err != nil {
		log.Println(err)
	}
	defer resp.Body.Close()

	err = os.MkdirAll(filepath.Base(filename), 0700)
	if err != nil {
		log.Println(err)
		return
	}
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0600)

	if err != nil {
		log.Println(err)
	}
	defer f.Close()
	io.Copy(f, resp.Body)
}

func processHtml(node *html.Node, found *[]string) bool {
	isEnd := false

	if node.Type == html.ElementNode && node.Data == "a" {
		links := node.Attr
		if len(links) != 1 {
			log.Println(links)
		}

		key, val := links[0].Key, links[0].Val
		if key != "href" {
			log.Fatal(key)
		}
		val = strings.TrimRight(val, "/")
		if strings.HasSuffix(val, ".fits") {
			isEnd = true
		}

		if _, err := strconv.Atoi(val); err != nil && !isEnd {
			return false
		}
		*found = append(*found, val)
	}

	for c := node.FirstChild; c != nil; c = c.NextSibling {
		isEnd = processHtml(c, found) || isEnd
	}

	return isEnd
}

func test_process() {
	r, _ := http.Get("http://archive.stsci.edu/pub/kepler/lightcurves/0007/000757137/")
	node, _ := html.Parse(r.Body)
	var test []string
	processHtml(node, &test)
	fmt.Println(test)
}

func main() {

	searchDownload(rootHtml)
	// try failed again
	for !failed.Empty() {
		fmt.Printf("trying failed... (current failed number): %d\n", failed.Size())
		wg.Add(1)
		go func() {
			defer wg.Done()
			searchDownload(failed.PopFront().(string))
		}()
	}

	wg.Wait()
	fmt.Println("res", subRoots)
	// test_process()
}
