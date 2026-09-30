# Bundled data

## subdomains-top20000.txt

The default DNS wordlist used by `--fuzz` when no `--wordlist` is given (and
`$SUBSEER_WORDLIST` is unset). It is a verbatim copy of
`Discovery/DNS/subdomains-top1million-20000.txt` from **SecLists**
(https://github.com/danielmiessler/SecLists), redistributed here under the MIT
License:

```
MIT License

Copyright (c) 2012-2024 Daniel Miessler, Jason Haddix, g0tmi1k, and SecLists contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

To refresh it, copy the file of the same name from a current SecLists checkout.
