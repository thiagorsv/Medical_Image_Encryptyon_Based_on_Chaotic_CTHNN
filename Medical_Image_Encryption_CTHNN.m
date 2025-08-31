close all; clc;

%% keys
h = 0.01; w12 = -2; w13 = -7; N = 100; x01 = 0.01; x02 = 0.01; x03 = 0; 
image_encryption_arnold_hnn(h, w12, w13, N, x01, x02, x03);

function image_encryption_arnold_hnn(h, w12, w13, num_iter, x01, x02, x03)
    %% Load image
    I = imread('medica3.PNG'); % File name

    Igray = rgb2gray(I);

    [M,N] = size(Igray);
    P = max(M,N);
    Ipad = uint8(zeros(P,P));
    Ipad(1:M, 1:N) = Igray;
    I = Ipad;


    if size(I,3)==3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    [M,N] = size(Igray);
    assert(M==N, 'Arnold Cat Map requiere imagen cuadrada.');

    %%Arnold Cat Map
    Permuted = arnold_cat_map(Igray, num_iter);


    Npix = M*N;
    init = [x01; x02; x03]; % Initial State
    dt = h;

    traj = hnnRK4(init, dt, Npix, w12, w13);
    zseq = traj(1,:);
    znorm = round((zseq - min(zseq)) / (max(zseq)-min(zseq)) * 255);
    mask = uint8(znorm);

    %% XOR
    Pvec = Permuted(:);
    EncryptedVec = bitxor(Pvec, mask');
    Encrypted = reshape(EncryptedVec, M, N);

    %% show
    figure;
    subplot(2,2,1); imshow(Igray); title('Original');
    subplot(2,2,2); imshow(Permuted); title('Shuffled (Arnold Cat)');
    subplot(2,2,3); imshow(Encrypted); title('Encrypted');

    %%
    DecryptedVec = bitxor(EncryptedVec, mask');
    DePermuted = reshape(DecryptedVec, M, N);

    
    Recovered = inverse_arnold_cat_map(DePermuted, num_iter);

    subplot(2,2,4); imshow(Recovered); title('Decrypted');

    %fprintf('Píxeles diferentes: %d\\n', sum(abs(double(Igray(:)) - double(Recovered(:))) > 0));


% histogram
figure(2);
subplot(2,3,1); imshow(Igray); title('Original');
subplot(2,3,2); imshow(Permuted); title('Shuffled');
subplot(2,3,3); imshow(Encrypted);  title('Encrypted');
subplot(2,3,4); imhist(Igray); 
subplot(2,3,5); imhist(Permuted); 
subplot(2,3,6); imhist(Encrypted);

end

%% === Arnold Cat Map ===
function out = arnold_cat_map(I, num_iter)
    N = size(I,1);
    out = I;
    for k = 1:num_iter
        tmp = zeros(N,N,'uint8');
        for x = 0:N-1
            for y = 0:N-1
                x_p = mod(x + y, N);
                y_p = mod(x + 2*y, N);
                tmp(x_p+1, y_p+1) = out(x+1, y+1);
            end
        end
        out = tmp;
    end
end

%% === Inverse Arnold Cat Map ===
function out = inverse_arnold_cat_map(I, num_iter)
    N = size(I,1);
    out = I;
    for k = 1:num_iter
        tmp = zeros(N,N,'uint8');
        for x = 0:N-1
            for y = 0:N-1
                x_p = mod(2*x - y, N);
                y_p = mod(-x + y, N);
                tmp(x_p+1, y_p+1) = out(x+1, y+1);
            end
        end
        out = tmp;
    end
end

%% === HNN ===
function traj = hnnRK4(init, dt, steps, w12, w13)
    traj = zeros(3, steps);
    x = init(1); y = init(2); z = init(3);

    for k = 1:steps
        traj(:,k) = [x; y; z];
        f = @(u) [ -u(1)+w12*tanh(u(2))+w13*tanh(u(3)); ...
                   -u(2)+1*tanh(u(1))+3*tanh(u(2))+3*tanh(u(3)); ...
                   -u(3)-1*tanh(u(1))-3*tanh(u(2)) ];
        k1 = f([x;y;z]);
        k2 = f([x;y;z] + 0.5*dt*k1);
        k3 = f([x;y;z] + 0.5*dt*k2);
        k4 = f([x;y;z] +     dt*k3);
        delta = (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        x = x + delta(1); y = y + delta(2); z = z + delta(3);
    end
end