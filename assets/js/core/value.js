// Scalar automatic differentiation (the micrograd idea), for week 4.
// Every Value remembers which values made it and how to pass its gradient back.
// backward() visits the graph in reverse topological order applying the chain rule.

let nextId = 0;

export class Value {
  constructor(data, children = [], op = '', label = '') {
    this.id = nextId++;
    this.data = data;
    this.grad = 0;
    this.children = children;
    this.op = op;
    this.label = label;
    this._backward = () => {};
  }

  static of(x) {
    return x instanceof Value ? x : new Value(x);
  }

  add(other) {
    const o = Value.of(other);
    const out = new Value(this.data + o.data, [this, o], '+');
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other) {
    const o = Value.of(other);
    const out = new Value(this.data * o.data, [this, o], '×');
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  pow(k) {
    const out = new Value(this.data ** k, [this], `^${k}`);
    out._backward = () => {
      this.grad += k * this.data ** (k - 1) * out.grad;
    };
    return out;
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    return this.add(Value.of(other).neg());
  }

  div(other) {
    return this.mul(Value.of(other).pow(-1));
  }

  exp() {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], 'exp');
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log() {
    const out = new Value(Math.log(this.data), [this], 'log');
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  tanh() {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu() {
    const out = new Value(Math.max(0, this.data), [this], 'ReLU');
    out._backward = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  /** Nodes in topological order (inputs first). */
  topo() {
    const order = [];
    const seen = new Set();
    const visit = (v) => {
      if (seen.has(v)) return;
      seen.add(v);
      v.children.forEach(visit);
      order.push(v);
    };
    visit(this);
    return order;
  }

  backward() {
    const order = this.topo();
    order.forEach((v) => (v.grad = 0));
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) order[i]._backward();
  }
}

/** Softmax over Values (stable: subtract the max first). */
export function softmaxV(logits) {
  const max = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(max).exp());
  const sum = exps.reduce((a, b) => a.add(b));
  return exps.map((e) => e.div(sum));
}

/** Cross-entropy: -log p(target). */
export function crossEntropyV(logits, target) {
  return softmaxV(logits)[target].log().neg();
}
