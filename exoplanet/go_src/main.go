package main

import (
	"fmt"
	"os"
)

func main() {
	abs := `C:\Users\User\dev\data\train\0007\0007570s99`
	if _, err := os.Stat(abs); err == nil {
		fmt.Println("exists")
	} else {
		fmt.Println("not exists")
	}
}
