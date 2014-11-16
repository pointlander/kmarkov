package main

import (
	"bytes"
	"fmt"
	"github.com/nfnt/resize"
	"github.com/pointlander/compress"
	"github.com/thoj/go-galib"
	"image"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
)

var data []byte
var dataComplexity float64

func normalize(d *[256][256]uint32) {
	for i := range d {
		var total, sum uint64
		for ii := range d[i] {
			total += uint64(d[i][ii])
		}

		if total > 0 {
			for ii := range d[i] {
				sum += uint64(d[i][ii])
				d[i][ii] = uint32((sum * math.MaxUint32) / total)
			}
		}
	}
}

func kc(a []byte) float64 {
	input, in, output := make([]byte, len(a)), make(chan []byte, 1), &bytes.Buffer{}
	copy(input, a)
	in <- input
	close(in)
	compress.BijectiveBurrowsWheelerCoder(in).MoveToFrontRunLengthCoder().AdaptiveCoder().Code(output)
	return float64(output.Len())
}

func fitness(g *ga.GAFixedBitstringGenome) float64 {
	var a, b [256][256]uint32
	var c byte
	for i, j := range g.Gene {
		pixel := data[i]
		if j {
			a[c][pixel]++
		} else {
			b[c][pixel]++
		}
		c = pixel
	}

	normalize(&a)
	normalize(&b)

	c = 0
	model := make([]byte, len(data))
	for i, j := range g.Gene {
		if j {
			r, pixel := rand.Uint32(), byte(0)
			for i := range a[c] {
				if r < a[c][i] {
					pixel = byte(i)
					break
				}
			}
			model[i], c = pixel, pixel
		} else {
			r, pixel := rand.Uint32(), byte(0)
			for i := range b[c] {
				if r < b[c][i] {
					pixel = byte(i)
					break
				}
			}
			model[i], c = pixel, pixel
		}
	}

	modelComplexity, complexity := kc(model), kc(append(model, data...))
	ncd := (complexity - math.Min(dataComplexity, modelComplexity)) / math.Max(dataComplexity, modelComplexity)
	return ncd
}

func main() {
	runtime.GOMAXPROCS(64)
	fmt.Println("kmarkov")

	in, err := os.Open("lenna.png")
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()

	img, err := png.Decode(in)
	if err != nil {
		log.Fatal(err)
	}

	bounds, scale := img.Bounds(), uint(2)
	img = resize.Resize(uint(bounds.Max.X) / scale, 0, img, resize.NearestNeighbor)
	bounds = img.Bounds()
	fmt.Printf("%vx%v\n", bounds.Max.X, bounds.Max.Y)

	small, err := os.Create("lenna_small.png")
	if err != nil {
		log.Fatal(err)
	}
	defer small.Close()
	err = png.Encode(small, img)
	if err != nil {
		log.Fatal(err)
	}

	data = make([]byte, bounds.Max.X * bounds.Max.Y)
	for y := 0; y < bounds.Max.Y; y++ {
		offset := y * bounds.Max.X
		for x := 0; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			data[offset + x] = byte((r + g + b) / (3 * 256))
		}
	}
	dataComplexity = kc(data)

	gray := image.NewGray(bounds)
	gray.Pix = data
	out, err := os.Create("lenna_gray.png")
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	err = png.Encode(out, gray)
	if err != nil {
		log.Fatal(err)
	}

	mutator := ga.NewMultiMutator()
	//mutator.Add(new(ga.GAShiftMutator))
	mutator.Add(new(ga.GASwitchMutator))

	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.7, 5),
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     mutator,
		PMutate:     0.8,
		PBreed:      0.8}

	genetic := ga.NewGAParallel(param, 2)
	genome := ga.NewFixedBitstringGenome(make([]bool, bounds.Max.X * bounds.Max.Y), fitness)
	genetic.Init(10, genome)
	genetic.Optimize(2048)

	best := genetic.Best().(*ga.GAFixedBitstringGenome)
	fmt.Printf("best=%v\n", best.Score())

	mask, maskPixels := image.NewGray(bounds), make([]byte, bounds.Max.X * bounds.Max.Y)
	for i, j := range best.Gene {
		if j {
			maskPixels[i] = 0
		} else {
			maskPixels[i] = 255
		}
	}
	mask.Pix = maskPixels
	outMask, err := os.Create("lenna_mask.png")
	if err != nil {
		log.Fatal(err)
	}
	defer outMask.Close()
	err = png.Encode(outMask, mask)
	if err != nil {
		log.Fatal(err)
	}
}
