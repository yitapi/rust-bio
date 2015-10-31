use stats::Prob;


pub fn viterbi<T>(observations: &[T], hmm: &HMM<T>) {
    let mut V: Vec<Vec<usize>> = vec![Vec::new()];
    let mut path: Vec<Vec<usize>> = Vec::new();

    // initialize base cases (t = 0)
    {
        let obs = observations[0];
        for y in hmm.states() {
            V[0].push(hmm.init_prob(y) * hmm.emit_prob(y, obs));
            path.push(vec![y]);
        }
    }

    // t > 0
    for (t, obs) in observations.iter().enumerate().skip(1) {
        let v_last = V[t];
        let mut v = Vec::new();
        let mut newpath = Vec::new();

        for y in hmm.states() {
            let (prob, state) = hmm.states().map(|y0| (v_last[y0] * hmm.trans_prob(y0, y) * hmm.emit_prob(y, obs), y0)).max_by(|(prob, state)| prob);
            v.push(prob);
            let mut p = Vec::new();
            p.extend(path[state].iter());
            p.push(y);
            newpath.push(p);
        }
        path = newpath;
    }

}
