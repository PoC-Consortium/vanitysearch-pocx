use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pocx_vanity::crypto::{self, Crypto, bech32_utils};

fn bench_hash160(c: &mut Criterion) {
    let pubkey = hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798").unwrap();
    
    c.bench_function("hash160", |b| {
        b.iter(|| crypto::hash160(black_box(&pubkey)))
    });
}

fn bench_bech32_encode(c: &mut Criterion) {
    let hash160: [u8; 20] = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6")
        .unwrap()
        .try_into()
        .unwrap();
    
    c.bench_function("bech32_encode", |b| {
        b.iter(|| bech32_utils::encode(black_box(&hash160), true))
    });
}

fn bench_wif_generation(c: &mut Criterion) {
    let crypto = Crypto::new();
    let priv_key: [u8; 32] = [1u8; 32];
    
    c.bench_function("wif_generation", |b| {
        b.iter(|| crypto.to_wif(black_box(&priv_key), true))
    });
}

fn bench_public_key(c: &mut Criterion) {
    let crypto = Crypto::new();
    let priv_key: [u8; 32] = [1u8; 32];
    
    c.bench_function("public_key_generation", |b| {
        b.iter(|| crypto.public_key(black_box(&priv_key)))
    });
}

criterion_group!(
    benches,
    bench_hash160,
    bench_bech32_encode,
    bench_wif_generation,
    bench_public_key,
);
criterion_main!(benches);
