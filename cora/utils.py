import torch
from torchvision.utils import save_image
import cv2
import math
import matplotlib.pyplot as plt


def hdr_to_tensor(img_path, pmap_h, pmap_w):
    img = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH)
    img = cv2.resize(img, (pmap_w, pmap_h))
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0]]


def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def normlizeSH(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / (4 * math.pi * math.factorial(l + m)))


def SH(l, m, theta, phi):
    if m == 0:
        return normlizeSH(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    elif m > 0:
        return math.sqrt(2.0) * normlizeSH(l, m) * \
                torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * normlizeSH(l, -m) * \
                torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))


def SH_xyz(l, m, x, y, z):
    cos_theta = z
    theta = torch.acos(cos_theta)
    phi = torch.atan2(y, x)
    if m == 0:
        return normlizeSH(l, m) * associated_legendre_polynomial(l, m, z)
    elif m > 0:
        return math.sqrt(2.0) * normlizeSH(l, m) * \
                torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * normlizeSH(l, -m) * \
                torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))


def pm2sh(pm, order=3):
    '''
    input: pm [b,3,h,w] and w=2h
    output: 
        coeffs: with size of [b,3,9]
        pm_sh: [b,3,h,w]
    '''
    _, _, h, w = pm.size()
    
    theta = torch.linspace(0, math.pi, h).to(pm.device)  # [h] from 0 to pi
    phi = torch.linspace(0, 2 * math.pi, w).to(pm.device)  # [w] from 0 to 2pi
    theta = theta[..., None].repeat(1, w)  # [h,w]
    phi = phi[None, ...].repeat(h, 1)  # [h,w]

    dphi = 2 * math.pi / w
    dtheta = math.pi / h
    
    # calculate integral
    pm = pm[..., None]  # [b,3,h,w,1]

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]
    sin_theta = torch.sin(theta).unsqueeze(-1)  # [h,w,1]
    coeffs = torch.sum(pm * sh_basis * sin_theta * dtheta * dphi, dim=(2, 3))  # [b,3,n]

    # get pm represented by sh
    coeffs_ = coeffs[:, :, None, None, :]  # [b,3,1,1,n]
    pm_sh = torch.sum(coeffs_ * sh_basis, dim=-1)
    
    return coeffs, pm_sh


def save_hdr_image(img, save_path, gamma=1/1.8):
    save_image(torch.pow(img / (1.0 + img), gamma), save_path)


def gen_sh(hdri_path, order):
    '''
    cur_coeffs: [3,order**2]
    pmap: [3,h,2h]
    '''
    pmap = hdr_to_tensor(hdri_path, 224, 448)
    pmap = pmap.cuda()
    cur_coeffs, pm_sh = pm2sh(pmap, order=order)  # [1,3,order**2]
    return cur_coeffs[0], pmap[0]


def bp2sh(order=3, shiness=64, device="cuda"):
    '''
    coeffs: [order**2]
    '''
    h = 512
    w = 2 * h
    
    theta = torch.linspace(0, math.pi, h).to(device)  # [h] from 0 to pi
    phi = torch.linspace(0, 2 * math.pi, w).to(device)  # [w] from 0 to 2pi
    theta = theta[..., None].repeat(1, w)  # [h,w]
    phi = phi[None, ...].repeat(h, 1)  # [h,w]

    dphi = 2 * math.pi / w
    dtheta = math.pi / h
    
    # calculate integral
    lobe = (torch.cos(theta / 2)) ** shiness
    mask = torch.cos(theta) > 0
    lobe = lobe * mask.float()

    lobe = lobe * (shiness + 2) / (4 * math.pi * (2 - 2 ** (-shiness / 2)))
    lobe = lobe.unsqueeze(-1)

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi) * math.sqrt(4 * math.pi / (2 * l + 1)))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]
    sin_theta = torch.sin(theta).unsqueeze(-1)  # [h,w,1]
    coeffs = torch.sum(lobe * sh_basis * sin_theta * dtheta * dphi, dim=(0, 1))  # [n]
    
    return coeffs


def diffuse2sh(order=3, device="cuda"):
    '''
    coeffs: [order**2]
    '''
    h = 512
    w = 2 * h
    
    theta = torch.linspace(0, math.pi, h).to(device)  # [h] from 0 to pi
    phi = torch.linspace(0, 2 * math.pi, w).to(device)  # [w] from 0 to 2pi
    theta = theta[..., None].repeat(1, w)  # [h,w]
    phi = phi[None, ...].repeat(h, 1)  # [h,w]

    dphi = 2 * math.pi / w
    dtheta = math.pi / h
    
    # calculate integral
    lobe = torch.cos(theta)
    mask = torch.cos(theta) > 0
    lobe = lobe * mask.float()

    lobe = lobe / math.pi
    lobe = lobe.unsqueeze(-1)

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi) * math.sqrt(4 * math.pi / (2 * l + 1)))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]
    sin_theta = torch.sin(theta).unsqueeze(-1)  # [h,w,1]
    coeffs = torch.sum(lobe * sh_basis * sin_theta * dtheta * dphi, dim=(0, 1))  # [n]
    
    return coeffs


def plot_coeffs(coeffs, save_path):
    order = int(math.sqrt(len(coeffs)))
    cnt = 0
    x, y = [], []
    for l in range(order):
        for m in range(-l, l + 1):
            if m == 0: 
                x.append(l)
                y.append(coeffs[cnt].item())
            cnt += 1
    plt.plot(x, y)
    plt.savefig(save_path)
    plt.cla()


if __name__ == "__main__":
    # hdri_path = "/root/autodl-tmp/HDRI/misty_dawn.hdr"
    # cur_coeffs, pmap = gen_sh(hdri_path, order=9)
    # save_hdr_image(pmap, "env.png")
    diff_coeffs = diffuse2sh(order=9)
    bp_coeffs = bp2sh(order=9, shiness=64)
    plot_coeffs(diff_coeffs, "diff.png")
    plot_coeffs(bp_coeffs, "bp.png")
