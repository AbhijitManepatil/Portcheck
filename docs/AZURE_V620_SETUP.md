# Running on an Azure NGads V620 VM (AMD Radeon PRO V620)

Notes from the real-GPU test on 2026-11-12 (`release-evidence/v0.1-azure-amd-v620/`).
Microsoft documents ROCm support for this VM series but, when this was written, had no
Linux driver guide for it; its guide covers the similar NVads V710 series. The steps below
are what worked once, on one VM. Treat them as a starting point.

Tested with: `Standard_NG8ads_V620_v1`, Ubuntu 22.04.5, kernel `6.8.0-1070-azure`,
Secure Boot off (the VM was created with security type `Standard`, so the out-of-tree
kernel module can load).

## 1. Kernel headers, extra modules and firmware

```bash
sudo apt-get update
sudo apt-get install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" linux-firmware
```

The Ubuntu cloud image blocklists `amdgpu` in
`/etc/modprobe.d/blacklist-radeon-instinct.conf`. The steps below load it by hand; remove
that entry and run `sudo update-initramfs -uk all` if you want it loaded at boot.

## 2. AMD's DKMS driver

The kernel's own `amdgpu` module does not initialise this virtual GPU. Install AMD's:

```bash
cd /tmp
curl -fsSLO https://repo.radeon.com/amdgpu-install/6.4.4/ubuntu/jammy/amdgpu-install_6.4.60404-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.4.60404-1_all.deb
sudo apt-get update
sudo apt-get install -y amdgpu-dkms
```

## 3. Firmware from the Azure driver build

With the firmware from step 2 the driver stops with
`failed to load ucode CAP(0x0)` and `PSP firmware loading failed`. The Azure host expects a
different `sienna_cichlid_cap.bin`. The driver build that Microsoft's V710 guide installs
has it. Take the firmware package from that repository (use the `repo.radeon.com` path
given in Microsoft's guide,
`articles/virtual-machines/linux/azure-n-series-amd-gpu-driver-linux-installation-guide.md`
in `MicrosoftDocs/azure-compute-docs`), unpack it, and copy its `sienna_cichlid_*` files
over the installed ones:

```bash
# REPO is the amdgpu/6.4.2.2/ubuntu path under the repository named in Microsoft's guide
curl -fsSL "$REPO/pool/main/a/amdgpu-dkms/" | grep -o 'amdgpu-dkms-firmware_[^"<]*_all\.deb' | sort -u | tail -1
curl -fsSLO "$REPO/pool/main/a/amdgpu-dkms/<that file>"
dpkg-deb -x amdgpu-dkms-firmware_*_all.deb fw
sudo cp fw/lib/firmware/updates/amdgpu/sienna_cichlid_* /lib/firmware/updates/amdgpu/
sudo reboot
```

Only `sienna_cichlid_cap.bin` and `sienna_cichlid_vcn.bin` differed from the public
package.

## 4. Load and check

```bash
sudo modprobe amdgpu
ls /dev/kfd /dev/dri/renderD128
sudo dmesg | grep -E "kfd|amdgpu" | tail
```

Expect `kfd kfd: amdgpu: added device 1002:73ae` and both device files.

## 5. PyTorch and portcheck

The ROCm 10.0 wheels need Python 3.11 or newer; Ubuntu 22.04 ships 3.10, so use `uv` (or
another way to get a newer Python):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch --index-url https://download.pytorch.org/whl/rocm10.0
uv pip install --python venv/bin/python /path/to/Portcheck transformers
venv/bin/portcheck fingerprint
```

The fingerprint should say `vendor: amd`. Run as root, or add your user to the `render`
and `video` groups.

The first capture on the GPU is slow (several minutes for the fixture workload).

## 6. In a container

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --group-add "$(getent group render | cut -d: -f3)" <image> ...
```

The default `target.rocm_image` is about 52 GB unpacked; give the OS disk at least 128 GB.
