use anyhow::Result;
use super::Tile;

fn get_index(position: (usize, usize), size: (usize, usize)) -> usize {
    position.0 + position.1 * size.0
}

pub fn generate(size: (usize, usize)) -> Result<Vec<Tile>> {
    let mut wavefunction: Vec<u8> = vec![0xf; size.0 * size.1];

    loop {
        if !collapse_min(&mut wavefunction, size) {
            break;
        }

        enforce_constraints(&mut wavefunction, size)
    }

    wavefunction
        .iter()
        .map(|&set| {
            Ok(match set {
                0x1 => Tile::Water,
                0x2 => Tile::Beach,
                0x4 => Tile::Grass,
                0x8 => Tile::Forest,
                _ => anyhow::bail!("Wavefunction not fully collapsed"),
            })
        })
        .collect::<Result<Vec<_>>>()
}

#[allow(dead_code)]
fn collapse(wavefunction: &mut [u8], size: (usize, usize)) -> bool {
    let mut count = 0;
    let mut collapse_pos = (usize::MAX, usize::MAX);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let options = wavefunction[get_index((x, y), size)].count_ones();
            if options > 1 {
                count += 1;
                if rand::random_range(0..count) < 1 {
                    collapse_pos = (x, y);
                }
            }
        }
    }

    if collapse_pos == (usize::MAX, usize::MAX) {
        return false;
    }

    let index = get_index(collapse_pos, size);
    let mut options = wavefunction[index];
    let chosen_option = rand::random_range(0..options.count_ones());
    for _ in 0..chosen_option {
        options &= options - 1;
    }
    options &= !(options - 1);
    wavefunction[index] = options;

    true
}

#[allow(dead_code)]
fn collapse_min(wavefunction: &mut [u8], size: (usize, usize)) -> bool {
    let mut min_options = 5;
    let mut count = 0;
    let mut collapse_pos = (usize::MAX, usize::MAX);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let options = wavefunction[get_index((x, y), size)].count_ones();
            if (2..min_options).contains(&options) {
                min_options = options;
                count = 1;
                collapse_pos = (x, y);
            } else if options == min_options {
                count += 1;
                if rand::random_range(0..count) < 1 {
                    collapse_pos = (x, y);
                }
            }
        }
    }

    if collapse_pos == (usize::MAX, usize::MAX) {
        return false;
    }

    let index = get_index(collapse_pos, size);
    let mut options = wavefunction[index];
    let chosen_option = rand::random_range(0..min_options);
    for _ in 0..chosen_option {
        options &= options - 1;
    }
    options &= !(options - 1);
    wavefunction[index] = options;

    true
}

fn enforce_constraints(wavefunction: &mut [u8], size: (usize, usize)) {
    let mut constraints = vec![0; size.0 * size.1];

    loop {
        for y in 0..size.1 {
            for x in 0..size.0 {
                let mut constraint = 0xf;
                if x > 0 {
                    let neighbor = wavefunction[get_index((x - 1, y), size)];
                    constraint &= (neighbor << 1) | neighbor | (neighbor >> 1);
                }
                if y > 0 {
                    let neighbor = wavefunction[get_index((x, y - 1), size)];
                    constraint &= (neighbor << 1) | neighbor | (neighbor >> 1);
                }
                if x < size.0 - 1 {
                    let neighbor = wavefunction[get_index((x + 1, y), size)];
                    constraint &= (neighbor << 1) | neighbor | (neighbor >> 1);
                }
                if y < size.1 - 1 {
                    let neighbor = wavefunction[get_index((x, y + 1), size)];
                    constraint &= (neighbor << 1) | neighbor | (neighbor >> 1);
                }

                constraints[get_index((x, y), size)] = constraint;
            }
        }

        let mut changed = false;
        for y in 0..size.1 {
            for x in 0..size.0 {
                let index = get_index((x, y), size);
                let options = wavefunction[index];
                let new_options = options & constraints[index];
                if new_options != options {
                    wavefunction[index] = new_options;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}
