#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <string>
#include <map>
#include <sstream>

// Eigenのインクルード
#include <Eigen/Dense>
#include <Eigen/Geometry>

// ==========================================
// 1. データ構造 (Eigenのベクトルを活用)
// ==========================================
struct Image3D {
	int nx, ny, nz;
	float sx, sy, sz;
	float ox, oy, oz;
	std::vector<float> data;

	Image3D(int x, int y, int z) : nx(x), ny(y), nz(z), sx(1), sy(1), sz(1), ox(0), oy(0), oz(0), data((size_t)x* y* z, 0.0f) {}

	// EigenのVector3fを受け取る補間関数
	inline float getPixel(const Eigen::Vector3f& p) const {
		int x0 = (int)std::floor(p.x()), y0 = (int)std::floor(p.y()), z0 = (int)std::floor(p.z());
		if (x0 < 0 || x0 >= nx - 1 || y0 < 0 || y0 >= ny - 1 || z0 < 0 || z0 >= nz - 1) return 0.0f;

		float dx = p.x() - x0, dy = p.y() - y0, dz = p.z() - z0;
		const float* d = data.data();
		auto sample = [&](int ix, int iy, int iz) { return d[((size_t)iz * ny + iy) * nx + ix]; };

		float c00 = sample(x0, y0, z0) * (1 - dx) + sample(x0 + 1, y0, z0) * dx;
		float c01 = sample(x0, y0, z0 + 1) * (1 - dx) + sample(x0 + 1, y0, z0 + 1) * dx;
		float c10 = sample(x0, y0 + 1, z0) * (1 - dx) + sample(x0 + 1, y0 + 1, z0) * dx;
		float c11 = sample(x0, y0 + 1, z0 + 1) * (1 - dx) + sample(x0 + 1, y0 + 1, z0 + 1) * dx;

		return (c00 * (1 - dy) + c10 * dy) * (1 - dz) + (c01 * (1 - dy) + c11 * dy) * dz;
	}
};

struct Timer {
	std::chrono::high_resolution_clock::time_point start;
	Timer() { start = std::chrono::high_resolution_clock::now(); }
	double elapsed() { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count(); }
};

// ==========================================
// 2. ユーティリティ
// ==========================================
Image3D loadFromMHD(const std::string& mhdPath) {
	std::ifstream ifs(mhdPath);
	if (!ifs) throw std::runtime_error("File not found: " + mhdPath);
	std::string line, key, dummy;
	std::map<std::string, std::string> meta;
	while (std::getline(ifs, line)) {
		if (line.find('=') == std::string::npos) continue;
		std::stringstream ss(line); ss >> key >> dummy;
		std::string val; std::getline(ss, val);
		size_t f = val.find_first_not_of(" \t\r\n"), l = val.find_last_not_of(" \t\r\n");
		if (f != std::string::npos) meta[key] = val.substr(f, l - f + 1);
	}
	int nx, ny, nz; std::stringstream ss(meta["DimSize"]); ss >> nx >> ny >> nz;
	Image3D vol(nx, ny, nz);
	std::stringstream ssS(meta["ElementSpacing"]); ssS >> vol.sx >> vol.sy >> vol.sz;
	std::stringstream ssO(meta["Offset"]); ssO >> vol.ox >> vol.oy >> vol.oz;
	std::string dir = mhdPath.substr(0, mhdPath.find_last_of("/\\") + 1);
	std::ifstream rfs(dir + meta["ElementDataFile"], std::ios::binary);
	if (meta["ElementType"] == "MET_SHORT") {
		std::vector<short> temp(vol.data.size()); rfs.read((char*)temp.data(), vol.data.size() * 2);
		for (size_t i = 0; i < temp.size(); ++i) vol.data[i] = (float)temp[i];
	}
	else rfs.read((char*)vol.data.data(), vol.data.size() * 4);
	float minV = vol.data[0], maxV = vol.data[0];
	for (float v : vol.data) { if (v < minV) minV = v; if (v > maxV) maxV = v; }
	for (float& v : vol.data) v = (v - minV) / (maxV - minV);
	return vol;
}

void saveAsMHD(const std::string& filename, const Image3D& image) {
	std::ofstream h(filename + ".mhd");
	h << "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix = 1 0 0 0 1 0 0 0 1\n";
	h << "Offset = " << std::fixed << std::setprecision(6) << image.ox << " " << image.oy << " " << image.oz << "\n";
	h << "CenterOfRotation = 0 0 0\nAnatomicalOrientation = RAI\nElementSpacing = " << image.sx << " " << image.sy << " " << image.sz << "\n";
	h << "DimSize = " << image.nx << " " << image.ny << " " << image.nz << "\nElementType = MET_FLOAT\nElementDataFile = " << filename << ".raw\n";
	std::ofstream r(filename + ".raw", std::ios::binary);
	r.write((const char*)image.data.data(), image.data.size() * 4);
}

Image3D resampleImage(const Image3D& moving, const Image3D& fixed, const float p[6]) {
	Image3D out(fixed.nx, fixed.ny, fixed.nz);
	out.sx = fixed.sx; out.sy = fixed.sy; out.sz = fixed.sz;
	out.ox = fixed.ox; out.oy = fixed.oy; out.oz = fixed.oz;

	// Eigenでの変換準備
	Eigen::Translation3f translation(-p[0], -p[1], -p[2]);
	Eigen::AngleAxisf roll(p[3], Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf pitch(p[4], Eigen::Vector3f::UnitY());
	Eigen::AngleAxisf yaw(p[5], Eigen::Vector3f::UnitZ());
	Eigen::Affine3f T = translation * yaw * pitch * roll;

	const Eigen::Vector3f center(fixed.ox + (fixed.nx - 1) * fixed.sx * 0.5f,
		fixed.oy + (fixed.ny - 1) * fixed.sy * 0.5f,
		fixed.oz + (fixed.nz - 1) * fixed.sz * 0.5f);
	const Eigen::Vector3f m_offset(moving.ox, moving.oy, moving.oz);
	const Eigen::Vector3f m_inv_spacing(1.0f / moving.sx, 1.0f / moving.sy, 1.0f / moving.sz);

#pragma omp parallel for collapse(2)
	for (int z = 0; z < fixed.nz; ++z) {
		for (int y = 0; y < fixed.ny; ++y) {
			for (int x = 0; x < fixed.nx; ++x) {
				// 1. 固定画像の物理座標
				Eigen::Vector3f p_f(x * fixed.sx + fixed.ox, y * fixed.sy + fixed.oy, z * fixed.sz + fixed.oz);

				// 2. 変換
				Eigen::Vector3f p_m = T * (p_f - center) + center;

				// 3. 移動画像のインデックス座標へ
				Eigen::Vector3f idx = (p_m - m_offset).cwiseProduct(m_inv_spacing);

				// 4. getPixelを新しい形式(ベクトル引数1つ)で呼び出し
				out.data[((size_t)z * fixed.ny + y) * fixed.nx + x] = moving.getPixel(idx);
			}
		}
	}
	return out;
}

// ==========================================
// 2. Eigen最適化 Mattes Registration
// ==========================================
class Mattes3DRegistrationAnalytic {
public:
	int bins = 32;

	inline float bspline(float x) {
		x = std::abs(x);
		if (x < 1.0f) return 0.666667f - x * x + 0.5f * x * x * x;
		if (x < 2.0f) return std::pow(2.0f - x, 3.0f) / 6.0f;
		return 0.0f;
	}

	// 輝度勾配計算（画像座標系での中央差分）
	Eigen::Vector3f getGradient(const Image3D& img, const Eigen::Vector3f& p) {
		float dx = (img.getPixel(p + Eigen::Vector3f(1, 0, 0)) - img.getPixel(p - Eigen::Vector3f(1, 0, 0))) * 0.5f;
		float dy = (img.getPixel(p + Eigen::Vector3f(0, 1, 0)) - img.getPixel(p - Eigen::Vector3f(0, 1, 0))) * 0.5f;
		float dz = (img.getPixel(p + Eigen::Vector3f(0, 0, 1)) - img.getPixel(p - Eigen::Vector3f(0, 0, 1))) * 0.5f;
		return Eigen::Vector3f(dx, dy, dz);
	}

	float computeMIAndGradient(const Image3D& fixed, const Image3D& moving, const float p[6], float grad[6], int step) {
		int threads = omp_get_max_threads();
		std::vector<std::vector<float>> hists(threads, std::vector<float>(bins * bins, 0.0f));
		// 各ヒストグラムビンごとに6次元の勾配ベクトルを蓄積
		std::vector<std::vector<Eigen::Matrix<float, 6, 1>>> gAccum(threads, std::vector<Eigen::Matrix<float, 6, 1>>(bins * bins, Eigen::Matrix<float, 6, 1>::Zero()));
		std::vector<float> totalWeights(threads, 0.0f);

		// 幾何変換の準備
		Eigen::Translation3f translation(-p[0], -p[1], -p[2]);
		Eigen::Matrix3f R = (Eigen::AngleAxisf(p[5], Eigen::Vector3f::UnitZ()) *
			Eigen::AngleAxisf(p[4], Eigen::Vector3f::UnitY()) *
			Eigen::AngleAxisf(p[3], Eigen::Vector3f::UnitX())).toRotationMatrix();
		Eigen::Affine3f T = translation * R;

		const Eigen::Vector3f center(fixed.ox + (fixed.nx - 1) * fixed.sx * 0.5f,
			fixed.oy + (fixed.ny - 1) * fixed.sy * 0.5f,
			fixed.oz + (fixed.nz - 1) * fixed.sz * 0.5f);
		const Eigen::Vector3f m_offset(moving.ox, moving.oy, moving.oz);
		const Eigen::Vector3f m_inv_spacing(1.0f / moving.sx, 1.0f / moving.sy, 1.0f / moving.sz);

		#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			// parallel と for を分けず、MSVCで最も安定する parallel for を使用
			// collapse が効かない場合でも z と y を一つにまとめて並列度を稼ぐ
			#pragma omp for
			for (int zy = 0; zy < (fixed.nz - 8) * (fixed.ny - 8); ++zy) {
				int z = 4 + (zy / (fixed.ny - 8)) * step;
				int y = 4 + (zy % (fixed.ny - 8)) * step;
				if (z >= fixed.nz - 4 || y >= fixed.ny - 4) continue;
				for (int x = 4; x < fixed.nx - 4; x += step) {
					Eigen::Vector3f pf(x * fixed.sx + fixed.ox, y * fixed.sy + fixed.oy, z * fixed.sz + fixed.oz);
					Eigen::Vector3f local_pf = pf - center;
					Eigen::Vector3f pm = T * local_pf + center;
					Eigen::Vector3f idx = (pm - m_offset).cwiseProduct(m_inv_spacing);

					float vF = fixed.data[((size_t)z * fixed.ny + y) * fixed.nx + x];
					float vM = moving.getPixel(idx);
					Eigen::Vector3f gM = getGradient(moving, idx);

					// ヤコビアンの構成
					Eigen::Matrix<float, 3, 6> J;
					J.block<3, 3>(0, 0) = -Eigen::Matrix3f::Identity();
					J.col(3) = R * Eigen::Vector3f::UnitX().cross(local_pf);
					J.col(4) = R * Eigen::Vector3f::UnitY().cross(local_pf);
					J.col(5) = R * Eigen::Vector3f::UnitZ().cross(local_pf);

					Eigen::Matrix<float, 6, 1> dVdp = J.transpose() * gM;

					float bF = vF * (bins - 5) + 2, bM = vM * (bins - 5) + 2;
					int iS = (int)bF - 1, jS = (int)bM - 1;
					for (int i = 0; i < 3; ++i) {
						float wF = bspline(bF - (iS + i));
						for (int j = 0; j < 3; ++j) {
							float wM = bspline(bM - (jS + j));
							float weight = wF * wM;
							hists[tid][(iS + i) * bins + (jS + j)] += weight;

							// B-Splineカーネルの微分近似
							float dwM = (bspline(bM - (jS + j) + 0.05f) - bspline(bM - (jS + j) - 0.05f)) * 10.0f;
							gAccum[tid][(iS + i) * bins + (jS + j)] += (wF * dwM) * dVdp;
							totalWeights[tid] += weight;
						}
					}
				}
			}
		}

		// --- MIと勾配の集計 ---
		float sumW = 0; for (float w : totalWeights) sumW += w;
		if (sumW < 1e-10) return 0;
		float invW = 1.0f / sumW;

		std::vector<float> pF(bins, 0), pM(bins, 0);
		std::vector<float> joint(bins * bins, 0);
		for (int t = 0; t < threads; ++t) {
			for (int k = 0; k < bins * bins; ++k) joint[k] += hists[t][k] * invW;
		}
		for (int i = 0; i < bins; ++i) {
			for (int j = 0; j < bins; ++j) {
				pF[i] += joint[i * bins + j];
				pM[j] += joint[i * bins + j];
			}
		}

		float mi = 0;
		Eigen::Matrix<float, 6, 1> totalGrad = Eigen::Matrix<float, 6, 1>::Zero();

		for (int i = 0; i < bins; ++i) {
			if (pF[i] < 1e-10) continue;
			for (int j = 0; j < bins; ++j) {
				float p_ij = joint[i * bins + j];
				if (p_ij > 1e-10 && pM[j] > 1e-10) {
					mi += p_ij * std::log(p_ij / (pF[i] * pM[j]));

					Eigen::Matrix<float, 6, 1> g_bin = Eigen::Matrix<float, 6, 1>::Zero();
					for (int t = 0; t < threads; ++t) g_bin += gAccum[t][i * bins + j];

					// MI勾配の公式: \sum (g_bin * (1 + log(p_ij/pM_j)))
					totalGrad += g_bin * (1.0f + std::log(p_ij / pM[j]));
				}
			}
		}

		for (int k = 0; k < 6; ++k) grad[k] = totalGrad[k] * invW;
		return mi;
	}

	// 1. 重心（Center of Mass）を計算する関数
	void initializeTransform(const Image3D& fixed, const Image3D& moving, float p[6]) {
		auto computeCOM = [](const Image3D& img) {
			double sumV = 0, sumX = 0, sumY = 0, sumZ = 0;
			for (int z = 0; z < img.nz; z += 2) { // 高速化のため2ボクセルおき
				for (int y = 0; y < img.ny; y += 2) {
					for (int x = 0; x < img.nx; x += 2) {
						float v = img.data[((size_t)z * img.ny + y) * img.nx + x];
						if (v > 0.1f) { // 背景ノイズを除去
							sumV += v;
							sumX += v * (x * img.sx + img.ox);
							sumY += v * (y * img.sy + img.oy);
							sumZ += v * (z * img.sz + img.oz);
						}
					}
				}
			}
			return Eigen::Vector3f(sumX / sumV, sumY / sumV, sumZ / sumV);
			};

		std::cout << "Auto-initializing offset via Center of Mass..." << std::endl;
		Eigen::Vector3f comF = computeCOM(fixed);
		Eigen::Vector3f comM = computeCOM(moving);

		// 重心の差を初期平行移動量としてセット (Rx, Ry, Rz は 0 のまま)
		p[0] = -(comM.x() - comF.x());
		p[1] = -(comM.y() - comF.y());
		p[2] = -(comM.z() - comF.z());

		std::cout << "Initial Offset: Tx=" << p[0] << ", Ty=" << p[1] << ", Tz=" << p[2] << std::endl;
	}

	// 2. 超高速化した optimize 関数
	void optimize(const Image3D& fixed, const Image3D& moving, float p[6]) {
		Timer timer;
		initializeTransform(fixed, moving, p);

		struct LevelConfig { int step; int maxIters; float stepSize; };
		// レベル設定をさらに攻撃的に（Stepを大きく、回数を動的に）
		std::vector<LevelConfig> configs = {
			{16, 8, 2.5f}, // 12回 → 8回
			{8,  6, 1.2f}, // 10回 → 6回
			{4,  5, 0.5f}  // 8回 → 5回
		};

		float lastMI = -1.0f;
		for (const auto& config : configs) {
			for (int iter = 0; iter < config.maxIters; ++iter) {
				float grad[6] = { 0 };
				float currentMI = computeMIAndGradient(fixed, moving, p, grad, config.step);

				// 収束判定: MIの変化が極小になったら次のレベルへ
				if (std::abs(currentMI - lastMI) < 1e-5) break;
				lastMI = currentMI;

				float gLen = 0;
				for (int i = 0; i < 6; ++i) gLen += grad[i] * grad[i];
				gLen = std::sqrt(gLen);
				if (gLen < 1e-9) break;

				// 更新（少し大きめのステップで加速）
				for (int i = 0; i < 6; ++i) {
					float scale = (i < 3) ? 1.0f : 0.012f;
					p[i] += config.stepSize * (grad[i] / gLen) * scale;
				}

				std::cout << "lastMI:" << lastMI << std::endl;

			}
		}
		std::cout << "\nLightning Fast Optimization Time: " << timer.elapsed() << "s" << std::endl;
	}

	void optimize_momentum(const Image3D& fixed, const Image3D& moving, float p[6]) {
		Timer timer;
		initializeTransform(fixed, moving, p);

		float velocity[6] = { 0, 0, 0, 0, 0, 0 };
		float gamma = 0.85f; // 慣性係数

		struct LevelConfig { int step; int maxIters; float stepSize; };
		std::vector<LevelConfig> configs = {
			{16, 8, 2.5f}, // 12回 → 8回
			{8,  6, 1.2f}, // 10回 → 6回
			{4,  5, 0.5f}  // 8回 → 5回
		};

		float lastMI = -1.0f;
		for (const auto& config : configs) {
			for (int iter = 0; iter < config.maxIters; ++iter) {
				float grad[6] = { 0 };
				float currentMI = computeMIAndGradient(fixed, moving, p, grad, config.step);

				if (std::abs(currentMI - lastMI) < 5e-6) break;
				lastMI = currentMI;

				float gLen = 0;
				for (int i = 0; i < 6; ++i) gLen += grad[i] * grad[i];
				gLen = std::sqrt(gLen + 1e-9f);

				for (int i = 0; i < 6; ++i) {
					float scale = (i < 3) ? 1.0f : 0.012f;
					float update = config.stepSize * (grad[i] / gLen) * scale;

					// Momentum 更新
					velocity[i] = gamma * velocity[i] + update;
					p[i] += velocity[i];
				}
			}
			// レベル切替時に慣性を少し減衰させて安定させる
			for (int i = 0; i < 6; ++i) velocity[i] *= 0.5f;
		}
		std::cout << "\nMomentum Optimization Time: " << timer.elapsed() << "s" << std::endl;
	}
};

// ==========================================
// 3. 精度評価用データ構造と関数
// ==========================================
struct RegistrationMetrics {
	double mi;
	double ncc;
	double mse;
	size_t overlapCount;

	void print() const {
		std::cout << "\n--- Registration Quality Metrics (Full Resolution) ---" << std::endl;
		std::cout << "MI (Mutual Information): " << std::fixed << std::setprecision(6) << mi << " (Higher is better)" << std::endl;
		std::cout << "NCC (Correlation):" << std::setprecision(6) << ncc << " (1.0 is perfect)" << std::endl;
		std::cout << "MSE (Mean Squared Error): " << std::setprecision(6) << mse << " (Lower is better)" << std::endl;
		std::cout << "Overlap Voxels:" << overlapCount << std::endl;
		std::cout << "------------------------------------------------------\n" << std::endl;
	}
};

// 最終的な位置合わせ精度を計算する (全ピクセル評価)
RegistrationMetrics computeFinalMetrics(const Image3D& fixed, const Image3D& moving, const float p[6]) {
	Timer timer;
	double sumF = 0, sumM = 0, sumFF = 0, sumMM = 0, sumFM = 0, sumSqErr = 0;
	size_t count = 0;

	Eigen::Translation3f translation(-p[0], -p[1], -p[2]);
	Eigen::Matrix3f R = (Eigen::AngleAxisf(p[5], Eigen::Vector3f::UnitZ()) *
		Eigen::AngleAxisf(p[4], Eigen::Vector3f::UnitY()) *
		Eigen::AngleAxisf(p[3], Eigen::Vector3f::UnitX())).toRotationMatrix();
	Eigen::Affine3f T = translation * R;

	const Eigen::Vector3f center(fixed.ox + (fixed.nx - 1) * fixed.sx * 0.5f,
		fixed.oy + (fixed.ny - 1) * fixed.sy * 0.5f,
		fixed.oz + (fixed.nz - 1) * fixed.sz * 0.5f);
	const Eigen::Vector3f m_offset(moving.ox, moving.oy, moving.oz);
	const Eigen::Vector3f m_inv_spacing(1.0f / moving.sx, 1.0f / moving.sy, 1.0f / moving.sz);

#pragma omp parallel for reduction(+:sumF,sumM,sumFF,sumMM,sumFM,sumSqErr,count)
	for (int z = 0; z < fixed.nz; ++z) {
		for (int y = 0; y < fixed.ny; ++y) {
			for (int x = 0; x < fixed.nx; ++x) {
				Eigen::Vector3f pf(x * fixed.sx + fixed.ox, y * fixed.sy + fixed.oy, z * fixed.sz + fixed.oz);
				Eigen::Vector3f pm = T * (pf - center) + center;
				Eigen::Vector3f idx = (pm - m_offset).cwiseProduct(m_inv_spacing);

				if (idx.x() < 0 || idx.x() >= moving.nx - 1 || idx.y() < 0 || idx.y() >= moving.ny - 1 || idx.z() < 0 || idx.z() >= moving.nz - 1)
					continue;

				float vF = fixed.data[((size_t)z * fixed.ny + y) * fixed.nx + x];
				float vM = moving.getPixel(idx);

				sumF += vF; sumM += vM;
				sumFF += (double)vF * vF; sumMM += (double)vM * vM;
				sumFM += (double)vF * vM;
				sumSqErr += (double)(vF - vM) * (vF - vM);
				count++;
			}
		}
	}

	RegistrationMetrics m = { 0, 0, 0, count };
	if (count > 0) {
		double meanF = sumF / count;
		double meanM = sumM / count;
		double varF = (sumFF / count) - (meanF * meanF);
		double varM = (sumMM / count) - (meanM * meanM);
		double covFM = (sumFM / count) - (meanF * meanM);
		m.ncc = (varF > 0 && varM > 0) ? covFM / (std::sqrt(varF) * std::sqrt(varM)) : 0;
		m.mse = sumSqErr / count;
	}

	std::cout << "\ncomputeFinalMetrics Time: " << timer.elapsed() << "s" << std::endl;

	return m;
}

// ==========================================
// 4. メイン（評価フロー統合版）
// ==========================================
int main() {
	try {
		Timer total;
		std::cout << "Loading images..." << std::endl;
		Image3D fixed = loadFromMHD("MRBrainTumor1.mhd");
		Image3D moving = loadFromMHD("MRBrainTumor2.mhd");

		Mattes3DRegistrationAnalytic reg;
		float p_res[6] = { 0, 0, 0, 0, 0, 0 };

		// 1. 最適化の実行
		//reg.optimize(fixed, moving, p_res);
		reg.optimize_momentum(fixed, moving, p_res);

		// 2. 最終精度の計算（全ピクセル・全指標）
		std::cout << "\nCalculating final registration accuracy..." << std::endl;

		// 全ピクセル（step=1）で厳密なMIを再計算
		float finalGrad[6] = { 0 };
		float finalMI = reg.computeMIAndGradient(fixed, moving, p_res, finalGrad, 2);

		// NCC, MSE, Overlap の計算
		RegistrationMetrics metrics = computeFinalMetrics(fixed, moving, p_res);
		metrics.mi = (double)finalMI;

		// 結果出力
		metrics.print();

		// 3. 結果画像の保存
		std::cout << "Generating final registered image..." << std::endl;
		Image3D registered = resampleImage(moving, fixed, p_res);
		saveAsMHD("moving_registered", registered);

		std::cout << "Registration Successful!" << std::endl;
		std::cout << "Final Parameters: Tx=" << p_res[0] << ", Ty=" << p_res[1] << ", Tz=" << p_res[2]
			<< " | Rx=" << p_res[3] << ", Ry=" << p_res[4] << ", Rz=" << p_res[5] << std::endl;
		std::cout << "Total Process Time: " << total.elapsed() << "s" << std::endl;
	}
	catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; }
	return 0;
}