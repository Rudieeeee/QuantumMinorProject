/**
 *
 * @param {number[][]} input 2D state array of the lights (each number is 0 or 1)
 * @returns {string[]}
 */
function generateConditions(input) {
  const conditions = [];
  const size = input.length;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let i = y * size + x;
      const adjacent = [];

      const offsets = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
      ];

      for (const [dx, dy] of offsets) {
        let [nx, ny] = [x + dx, y + dy];
        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;

        adjacent.push(ny * size + nx);
      }

      conditions.push([i, ...adjacent].map((v) => `map[${v}]`).join("^") + `===${input[y][x]}`);
    }
  }

  return conditions;
}

function solveConditions(conditions, size) {
  let solution = [];
  let minCount = Infinity;

  for (let i = 0; i < 2 ** (size ** 2); i++) {
    const map = new Array(size);
    for (let j = 0; j < size ** 2; j++) {
      map[j] = (i & (1 << j)) >> j;
    }
    let satisfies = true;
    for (const condition of conditions) {
      if (!eval(condition)) {
        satisfies = false;
        break;
      }
    }

    if (satisfies) {
      const count = map.reduce((a, c) => a + c, 0);
      if (count < minCount) {
        solution = map;
      }
    }
  }

  const niceSolution = [];
  let buffer = [];
  solution.forEach((v, i) => {
    if (i !== 0 && i % size === 0) {
      niceSolution.push(buffer);
      buffer = [];
    }
    buffer.push(v);
  });

  niceSolution.push(buffer);

  return niceSolution;
}

const SIZE = 5;

const conditions = generateConditions([
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 1],
  [0, 0, 0, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 1, 1, 0, 1],
]);

console.log(conditions);

console.log(solveConditions(conditions, SIZE));

// output:
// [
//   'map[0]^map[1]^map[5]===0',
//   'map[1]^map[2]^map[6]^map[0]===0',
//   'map[2]^map[3]^map[7]^map[1]===1',
//   'map[3]^map[4]^map[8]^map[2]===0',
//   'map[4]^map[9]^map[3]===0',
//   'map[5]^map[6]^map[10]^map[0]===0',
//   'map[6]^map[7]^map[11]^map[5]^map[1]===1',
//   'map[7]^map[8]^map[12]^map[6]^map[2]===1',
//   'map[8]^map[9]^map[13]^map[7]^map[3]===1',
//   'map[9]^map[14]^map[8]^map[4]===1',
//   'map[10]^map[11]^map[15]^map[5]===0',
//   'map[11]^map[12]^map[16]^map[10]^map[6]===0',
//   'map[12]^map[13]^map[17]^map[11]^map[7]===0',
//   'map[13]^map[14]^map[18]^map[12]^map[8]===0',
//   'map[14]^map[19]^map[13]^map[9]===0',
//   'map[15]^map[16]^map[20]^map[10]===0',
//   'map[16]^map[17]^map[21]^map[15]^map[11]===1',
//   'map[17]^map[18]^map[22]^map[16]^map[12]===1',
//   'map[18]^map[19]^map[23]^map[17]^map[13]===1',
//   'map[19]^map[24]^map[18]^map[14]===0',
//   'map[20]^map[21]^map[15]===0',
//   'map[21]^map[22]^map[20]^map[16]===1',
//   'map[22]^map[23]^map[21]^map[17]===1',
//   'map[23]^map[24]^map[22]^map[18]===0',
//   'map[24]^map[23]^map[19]===1'
// ]
// [
//   [ 0, 1, 1, 0, 0 ],
//   [ 1, 0, 1, 1, 0 ],
//   [ 1, 0, 0, 1, 0 ],
//   [ 0, 1, 0, 0, 1 ],
//   [ 0, 0, 0, 1, 1 ]
// ]
