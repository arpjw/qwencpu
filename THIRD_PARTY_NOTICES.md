# Third-party notices

This repository's 8-bit ISA, reference emulator, CPU circuit, dynamic ROM,
9B byte-CPU merge, and evaluation harness are developed in this repository.
They use the generic Qwen3.5 circuit IR, gates, scheduler, and weight emitter
from [Qwenputer](https://github.com/divagr18/qwenputer), pinned to commit
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. The file
`qwenputer-9b-port.patch` modifies Qwenputer's separate RV32 calculator
merge. Qwenputer's original project and RV32 design are by Divyansh Agrawal.

The Qwenputer source is licensed under the MIT License. Its copyright and
permission notice follow:

> MIT License
>
> Copyright (c) 2026 Divyansh Agrawal
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The Qwen3.5 pretrained weights are downloaded separately from
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). They are not
included here and retain their own model license and terms.
