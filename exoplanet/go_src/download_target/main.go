package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"golang.org/x/net/html"
)

const (
	rootHTML    = "http://archive.stsci.edu/pub/kepler/lightcurves"
	maxDownload = 100
)

var rootStore string

func init() {
	if runtime.GOOS == "windows" {
		rootStore = "C:/Users/User/dev/data/train/"
	} else if runtime.GOOS == "linux" {
		rootStore = "/home/chao/dev/data/train/"
	} else {
		log.Fatal("unknown system")
	}
}

var max = make(chan struct{}, maxDownload)
var wg sync.WaitGroup

func get_kepid(kepid_file string) []string {
	path, err := filepath.Abs(kepid_file)
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	var res []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		res = append(res, scanner.Text())
	}
	return res
}

func getURLFromKepID(kepids []string) []string {
	var res []string
	for _, kepid := range kepids {
		url := getURLFromOneID(kepid)
		res = append(res, url)
	}
	return res
}

func downloadTo(url, writeTo string) {
	max <- struct{}{}
	defer wg.Done()
	defer func() { <-max }()
	resp, err := http.Get(url)
	if err != nil {
		log.Println(err)
		return
	}
	defer resp.Body.Close()

	// ignore exists files

	if _, err = os.Stat(writeTo); err == nil {
		return
	}
	fmt.Printf("begin to download %q\n", url)

	f, err := os.OpenFile(writeTo, os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		log.Println(err)
		return
	}
	defer f.Close()
	io.Copy(f, resp.Body)

	fmt.Println("finished downloading")
}

func download(url string) {
	storeName := filepath.Join(rootStore, strings.TrimPrefix(url, rootHTML))
	err := os.MkdirAll(filepath.Dir(storeName), 0700)
	if err != nil {
		log.Println(err)
		return
	}
	downloadTo(url, storeName)
}

func getURLFromOneID(kepid string) string {
	return rootHTML + "/" + kepid[:4] + "/" + kepid
}

func parseURL(url string, action func(string)) {
	// fmt.Println("here", url)
	resp, err := http.Get(url)
	if err != nil {
		log.Println(err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		log.Printf("%q page not found\n", url)
		return
	}
	r, err := html.Parse(resp.Body)
	if err != nil {
		log.Println(err)
		return
	}

	processHTML(r, &url, action)
}

func processHTML(node *html.Node, url *string, action func(string)) {
	if node.Type == html.ElementNode && node.Data == "a" {
		key, val := node.Attr[0].Key, node.Attr[0].Val
		if key != "href" {
			log.Println("not hyper link", key)
			return
		}
		val = strings.TrimRight(val, "/")
		if strings.HasSuffix(val, "llc.fits") {
			// only download long cadence light curve
			wg.Add(1)
			// go download(*url + "/" + val)
			go action(*url + "/" + val)
			// fmt.Println(*url + "/" + val)
		}
	}
	for c := node.FirstChild; c != nil; c = c.NextSibling {
		processHTML(c, url, action)
	}
}

func main() {

	file := flag.String("file", "", "file containing target kepler id to download")
	id := flag.String("id", "", "kepler id to download")
	writeTo := flag.String("writeTo", "", "filename to write")
	flag.Parse()
	if *file != "" {
		kepid := get_kepid(*file)
		urls := getURLFromKepID(kepid)
		for _, url := range urls {
			parseURL(url, download)
		}
		wg.Wait()
		return
	}

	if *id != "" {
		url := getURLFromOneID(*id)
		if *writeTo == "" {
			log.Fatal("don't know where to write")
		}

		downloadToCurDir := func(url string) {
			fmt.Printf("writing to directory %q\n", *writeTo)
			if _, err := os.Stat(*writeTo); os.IsNotExist(err) {
				err = os.MkdirAll(*writeTo, 0700)
				if err != nil {
					log.Fatal(err)
				}
			}
			downloadTo(url, filepath.Join(*writeTo, filepath.Base(url)))
		}
		parseURL(url, downloadToCurDir)
		wg.Wait()
		return
	}
}
