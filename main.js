let isDrawing = false;
let oldPos = null;
let debug = false;
let lineWidth = 20;

function softmax(output) {
	let maximum = output.reduce((p, c) => {
		return p > c ? p : c;
	});
	let nominators = output.map((e) => {
		return Math.exp(e - maximum);
	});
	let denominator = nominators.reduce((p, c) => {
		return p + c;
	});
	let softmax = nominators.map((e) => {
		return e / denominator;
	});

	let maxIndex = 0;
	softmax.reduce((p, c, i) => {
		if (p < c) {
			maxIndex = i;
			return c;
		} else return p;
	});

	let result = [];

	for (let i = 0; i < output.length; i++) {
		if (i === maxIndex)
			result.push(1);
		else
			result.push(0);
	}

	return result;
}

const indexOfMax = (array) => {
	if (array.length === 0) {
		return -1;
	}

	let max = array[0];
	let maxIndex = 0;

	for (let i = 1; i < array.length; i++) {
		if (array[i] > max) {
			maxIndex = i;
			max = array[i];
		}
	}

	return maxIndex;
};

addEventListener('load', async () => {
	const canvas = document.getElementById('canvas');
	const demo = document.getElementById('demo');
	const result = document.getElementById('result');
	const ctx = canvas.getContext('2d');

	const urlSearchParams = new URLSearchParams(window.location.search);

	const net = new brain.NeuralNetwork();
	let nn;
	try {
		nn = await fetch(urlSearchParams.get('memory') || 'memory.json').then(response => response.json());
	} catch (e) {
		document.getElementById('error').classList.remove('error_hidden');
		document.getElementById('errorTitle').innerText = e.name;
		document.getElementById('errorDescription').innerText = e.message;
	}

	net.fromJSON(nn);

	document.querySelector('.loader').classList.add('loader_hidden');
	document.querySelector('.workspace__canvas').classList.remove('workspace__canvas_hidden');
	document.querySelector('.workspace__result').classList.remove('workspace__result_hidden');

	const update = () => {
		const gap = 5;
		const width = 28 - gap * 2, height = 28 - gap * 2;
		let input = cv.imread(canvas);
		const size = new cv.Size(width, height);

		cv.cvtColor(input, input, cv.COLOR_RGBA2GRAY, 0);
		const rect = cv.boundingRect(input);

		input = input.roi(rect);
		if (input.matSize[0] > input.matSize[1]) {
			const vertical = (input.matSize[0] - input.matSize[1]) / 2;
			cv.copyMakeBorder(input, input, 0, 0, vertical, vertical, cv.BORDER_CONSTANT);
		} else {
			const horizontal = (input.matSize[1] - input.matSize[0]) / 2;
			cv.copyMakeBorder(input, input, horizontal, horizontal, 0, 0, cv.BORDER_CONSTANT);
		}

		cv.resize(input, input, size, 0, 0, cv.INTER_AREA);
		cv.copyMakeBorder(input, input, gap, gap, gap, gap, cv.BORDER_CONSTANT);

		let demo = new cv.Mat();
		cv.resize(input, demo, new cv.Size(128, 128), 0, 0, cv.INTER_AREA);
		cv.imshow('demo', demo);

		const {data} = input;
		input.delete();

		const array = new Float32Array(data.length);
		data.forEach((value, index) => array[index] = value / 255);

		const output = net.run(array);

		if (debug) {
			console.log('NN Answer', output);
		}

		result.innerText = `${indexOfMax(softmax(output))}`;
	};

	const draw = (x, y) => {
		ctx.strokeStyle = '#f5f6fa';
		ctx.lineWidth = lineWidth;

		ctx.beginPath();
		ctx.moveTo(oldPos ? oldPos.x : x, oldPos ? oldPos.y : y);
		ctx.lineTo(x, y);
		ctx.stroke();

		update();

		oldPos = {x, y};
	};

	canvas.width = canvas.clientWidth;
	canvas.height = canvas.clientHeight;
	ctx.lineCap = 'round';

	addEventListener('resize', () => {
		canvas.width = canvas.clientWidth;
		canvas.height = canvas.clientHeight;
		ctx.lineCap = 'round';
	});

	addEventListener('keypress', ({code}) => {
		if (code === 'KeyD') {
			debug = !debug;
			demo.classList.toggle('demo_hidden');
		}
	});

	canvas.addEventListener('mousedown', ({offsetX, offsetY, button, target}) => {
		if (button !== 0 || target !== canvas) {
			return;
		}

		isDrawing = true;
		draw(offsetX, offsetY);

		document.querySelectorAll('.tooltip').forEach((tooltip) => {
			tooltip.classList.add('tooltip_hidden');
		});
	});
	addEventListener('mousemove', ({offsetX, offsetY, target}) => {
		if (target !== canvas) {
			return;
		}

		if (isDrawing) {
			draw(offsetX, offsetY);
		}
	});
	addEventListener('mouseup', () => {
		isDrawing = false;
		oldPos = null;
	});
	addEventListener('wheel', ({deltaY}) => {
		lineWidth = Math.min(Math.max(lineWidth + deltaY / Math.abs(deltaY) * -1, 1), 72);
		ctx.lineWidth = lineWidth;
	});
	addEventListener('contextmenu', (e) => {
		e.preventDefault();

		document.querySelectorAll('.tooltip').forEach((tooltip) => {
			tooltip.classList.remove('tooltip_hidden');
		});
		result.innerText = '';
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		demo.getContext('2d').clearRect(0, 0, demo.width, demo.height);
	});
});
