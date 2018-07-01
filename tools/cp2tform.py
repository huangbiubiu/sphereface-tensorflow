# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 06:54:28 2017

@author: Huang Yuheng
Reference: https://raw.githubusercontent.com/DQ0408/SphereFace-TensorFlow/master/matlab_cp2tform.py
"""

import numpy as np
import tensorflow as tf

"""
Introduction:
----------
TensorFlow implemetation form matlab function CP2TFORM(...)
with 'transformtype':
    1) 'nonreflective similarity'
    2) 'similarity'


MATLAB code:
----------
%--------------------------------------
% Function  findNonreflectiveSimilarity
%
function [trans, output] = findNonreflectiveSimilarity(uv,xy,options)
%
% For a nonreflective similarity:
%
% let sc = s*cos(theta)
% let ss = s*sin(theta)
%
%                   [ sc -ss
% [u v] = [x y 1] *   ss  sc
%                     tx  ty]
%
% There are 4 unknowns: sc,ss,tx,ty.
%
% Another way to write this is:
%
% u = [x y 1 0] * [sc
%                  ss
%                  tx
%                  ty]
%
% v = [y -x 0 1] * [sc
%                   ss
%                   tx
%                   ty]
%
% With 2 or more correspondence points we can combine the u equations and
% the v equations for one linear system to solve for sc,ss,tx,ty.
%
% [ u1  ] = [ x1  y1  1  0 ] * [sc]
% [ u2  ]   [ x2  y2  1  0 ]   [ss]
% [ ... ]   [ ...          ]   [tx]
% [ un  ]   [ xn  yn  1  0 ]   [ty]
% [ v1  ]   [ y1 -x1  0  1 ]
% [ v2  ]   [ y2 -x2  0  1 ]
% [ ... ]   [ ...          ]
% [ vn  ]   [ yn -xn  0  1 ]
%
% Or rewriting the above matrix equation:
% U = X * r, where r = [sc ss tx ty]'
% so r = X\\U.
%

K = options.K;
M = size(xy,1);
x = xy(:,1);
y = xy(:,2);
X = [x   y  ones(M,1)   zeros(M,1);
     y  -x  zeros(M,1)  ones(M,1)  ];

u = uv(:,1);
v = uv(:,2);
U = [u; v];

% We know that X * r = U
if rank(X) >= 2*K
    r = X \ U;
else
    error(message('images:cp2tform:twoUniquePointsReq'))
end

sc = r(1);
ss = r(2);
tx = r(3);
ty = r(4);

Tinv = [sc -ss 0;
        ss  sc 0;
        tx  ty 1];

T = inv(Tinv);
T(:,3) = [0 0 1]';

trans = maketform('affine', T);
output = [];

%-------------------------
% Function  findSimilarity
%
function [trans, output] = findSimilarity(uv,xy,options)
%
% The similarities are a superset of the nonreflective similarities as they may
% also include reflection.
%
% let sc = s*cos(theta)
% let ss = s*sin(theta)
%
%                   [ sc -ss
% [u v] = [x y 1] *   ss  sc
%                     tx  ty]
%
%          OR
%
%                   [ sc  ss
% [u v] = [x y 1] *   ss -sc
%                     tx  ty]
%
% Algorithm:
% 1) Solve for trans1, a nonreflective similarity.
% 2) Reflect the xy data across the Y-axis,
%    and solve for trans2r, also a nonreflective similarity.
% 3) Transform trans2r to trans2, undoing the reflection done in step 2.
% 4) Use TFORMFWD to transform uv using both trans1 and trans2,
%    and compare the results, Returnsing the transformation corresponding
%    to the smaller L2 norm.

% Need to reset options.K to prepare for calls to findNonreflectiveSimilarity.
% This is safe because we already checked that there are enough point pairs.
options.K = 2;

% Solve for trans1
[trans1, output] = findNonreflectiveSimilarity(uv,xy,options);


% Solve for trans2

% manually reflect the xy data across the Y-axis
xyR = xy;
xyR(:,1) = -1*xyR(:,1);

trans2r  = findNonreflectiveSimilarity(uv,xyR,options);

% manually reflect the tform to undo the reflection done on xyR
TreflectY = [-1  0  0;
              0  1  0;
              0  0  1];
trans2 = maketform('affine', trans2r.tdata.T * TreflectY);


% Figure out if trans1 or trans2 is better
xy1 = tformfwd(trans1,uv);
norm1 = norm(xy1-xy);

xy2 = tformfwd(trans2,uv);
norm2 = norm(xy2-xy);

if norm1 <= norm2
    trans = trans1;
else
    trans = trans2;
end
"""


class MatlabCp2tormException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


def tformfwd(trans: tf.Tensor, uv: tf.Tensor):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    # uv = np.hstack((
    #     uv, np.ones((uv.shape[0], 1))
    # ))
    # xy = np.dot(uv, trans)
    # xy = xy[:, 0:-1]

    with tf.name_scope("tformfwd"):
        uv = tf.concat(axis=1, values=[uv, tf.ones((uv.shape[0], 1))])
        xy = tf.tensordot(uv, trans, axes=(1, 0))
        xy = xy[:, 0:-1]

    return xy


def tforminv(trans: tf.Tensor, uv: tf.Tensor):
    """
    Function:
    ----------
        apply the inverse of affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed coordinates (x, y)
    """

    # Tinv = inv(trans)
    # xy = tformfwd(Tinv, uv)
    # return xy
    with tf.name_scope("tforminv"):
        Tinv = tf.matrix_inverse(trans)
        xy = tformfwd(Tinv, uv)
    return xy


def findNonreflectiveSimilarity(uv, xy, options=None):
    """
    Function:
    ----------
        Find Non-reflective Similarity Transform Matrix 'trans':
            u = uv[:, 0]
            v = uv[:, 1]
            x = xy[:, 0]
            y = xy[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    Parameters:
    ----------
        @uv: Kx2 np.array
            source points each row is a pair of coordinates (x, y)
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed
        @option: not used, keep it as None

    Returns:
        @trans: 3x3 np.array
            transform matrix from uv to xy
        @trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv

    Matlab:
    ----------
    % For a nonreflective similarity:
    %
    % let sc = s*cos(theta)
    % let ss = s*sin(theta)
    %
    %                   [ sc -ss
    % [u v] = [x y 1] *   ss  sc
    %                     tx  ty]
    %
    % There are 4 unknowns: sc,ss,tx,ty.
    %
    % Another way to write this is:
    %
    % u = [x y 1 0] * [sc
    %                  ss
    %                  tx
    %                  ty]
    %
    % v = [y -x 0 1] * [sc
    %                   ss
    %                   tx
    %                   ty]
    %
    % With 2 or more correspondence points we can combine the u equations and
    % the v equations for one linear system to solve for sc,ss,tx,ty.
    %
    % [ u1  ] = [ x1  y1  1  0 ] * [sc]
    % [ u2  ]   [ x2  y2  1  0 ]   [ss]
    % [ ... ]   [ ...          ]   [tx]
    % [ un  ]   [ xn  yn  1  0 ]   [ty]
    % [ v1  ]   [ y1 -x1  0  1 ]
    % [ v2  ]   [ y2 -x2  0  1 ]
    % [ ... ]   [ ...          ]
    % [ vn  ]   [ yn -xn  0  1 ]
    %
    % Or rewriting the above matrix equation:
    % U = X * r, where r = [sc ss tx ty]'
    % so r = X\\U.
    %
    """
    with tf.name_scope("findNonreflectiveSimilarity"):
        options = {'K': 2}

        K = options['K']
        M = xy.shape[0]
        # x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        x = tf.reshape(xy[:, 0], (-1, 1))
        # y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
        y = tf.reshape(xy[:, 1], (-1, 1))
        # print '--->x, y:\n', x, y

        # tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp1 = tf.concat(axis=1, values=(x, y, np.ones((M, 1)), np.zeros((M, 1))))
        # tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        tmp2 = tf.concat(axis=1, values=(y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        # X = np.vstack((tmp1, tmp2))
        X = tf.concat(axis=0, values=(tmp1, tmp2))
        # print '--->X.shape: ', X.shape
        # print 'X:\n', X

        # u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        u = tf.reshape(uv[:, 0], (-1, 1))
        # v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
        v = tf.reshape(uv[:, 1], (-1, 1))
        # U = np.vstack((u, v))
        U = tf.concat(axis=0, values=(u, v))
        # print '--->U.shape: ', U.shape
        # print 'U:\n', U

        # We know that X * r = U
        # if tf.rank(X) >= 2 * K:
        #     r, _, _, _ = tf.linalg.lstsq(X, U)
        #     r = tf.squeeze(r)
        # else:
        #     raise Exception('cp2tform:twoUniquePointsReq')
        r = tf.linalg.lstsq(X, U)
        r = tf.squeeze(r)

        # print '--->r:\n', r

        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        # Tinv = [[sc, -ss, 0],
        #         [ss, sc, 0],
        #         [tx, ty, 1]]
        Tinv = tf.transpose(tf.stack(axis=1, values=(tf.stack(axis=0, values=[sc, -ss, 0]),
                                                     tf.stack(axis=0, values=[ss, sc, 0]),
                                                     tf.stack(axis=0, values=[tx, ty, 1]))))

        # print '--->Tinv:\n', Tinv

        T = tf.matrix_inverse(Tinv)
        # print '--->T:\n', T

        # T[:, 2] = np.array([0, 0, 1])
        # T = tf.assign(ref=T[:, 2], value=[0, 0, 1]) Sliced assignment is only supported for variables
        T = tf.concat(axis=1, values=(T[:, :2], [[0], [0], [1]], T[:, 3:]))

    return T, Tinv


def findSimilarity(uv: tf.Tensor, xy: tf.Tensor, options=None):
    """
    Function:
    ----------
        Find Reflective Similarity Transform Matrix 'trans':
            u = uv[:, 0]
            v = uv[:, 1]
            x = xy[:, 0]
            y = xy[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    Parameters:
    ----------
        @uv: Kx2 np.array
            source points each row is a pair of coordinates (x, y)
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed
        @option: not used, keep it as None

    Returns:
    ----------
        @trans: 3x3 np.array
            transform matrix from uv to xy
        @trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv

    Matlab:
    ----------
    % The similarities are a superset of the nonreflective similarities as they may
    % also include reflection.
    %
    % let sc = s*cos(theta)
    % let ss = s*sin(theta)
    %
    %                   [ sc -ss
    % [u v] = [x y 1] *   ss  sc
    %                     tx  ty]
    %
    %          OR
    %
    %                   [ sc  ss
    % [u v] = [x y 1] *   ss -sc
    %                     tx  ty]
    %
    % Algorithm:
    % 1) Solve for trans1, a nonreflective similarity.
    % 2) Reflect the xy data across the Y-axis,
    %    and solve for trans2r, also a nonreflective similarity.
    % 3) Transform trans2r to trans2, undoing the reflection done in step 2.
    % 4) Use TFORMFWD to transform uv using both trans1 and trans2,
    %    and compare the results, Returnsing the transformation corresponding
    %    to the smaller L2 norm.

    % Need to reset options.K to prepare for calls to findNonreflectiveSimilarity.
    % This is safe because we already checked that there are enough point pairs.
    """
    with tf.name_scope("findSimilarity"):
        options = {'K': 2}

        #    uv = np.array(uv)
        #    xy = np.array(xy)

        # Solve for trans1
        trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

        # Solve for trans2

        # manually reflect the xy data across the Y-axis
        xyR = xy
        # xyR[:, 0] = -1 * xyR[:, 0]
        xyR0 = tf.reshape(xyR[:, 0], (-1, 1)) * -1
        xyR = tf.concat(axis=1, values=(xyR0, xyR[:, 1:]))

        trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

        # manually reflect the tform to undo the reflection done on xyR
        TreflectY = tf.constant([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=tf.float32)

        trans2 = tf.tensordot(trans2r, TreflectY, axes=(1, 0))

        # Figure out if trans1 or trans2 is better
        xy1 = tformfwd(trans1, uv)
        norm1 = tf.norm(xy1 - xy)

        xy2 = tformfwd(trans2, uv)
        # norm2 = norm(xy2 - xy)
        norm2 = tf.norm(xy2 - xy)

        # if norm1 <= norm2:
        #     return trans1, trans1_inv
        # else:
        #     trans2_inv = inv(trans2)
        #     return trans2, trans2_inv

    return tf.cond(norm1 <= norm2, lambda: (trans1, trans1_inv),
                   lambda: (trans2, tf.matrix_inverse(trans2)))


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'trans':
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        @reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
       @trans: 3x3 np.array
            transform matrix from uv to xy
        trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv
    """

    with tf.name_scope("get_similarity_transform"):
        if reflective:
            trans, trans_inv = findSimilarity(src_pts, dst_pts)
        else:
            trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)

    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans: tf.Tensor):
    """
    Function:
    ----------
        Convert Transform Matrix 'trans' into 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix from uv to xy

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    # cv2_trans = trans[:, 0:2].T
    with tf.name_scope("cvt_tform_mat_for_cv2"):
        cv2_trans = tf.transpose(trans[:, 0:2])

    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    with tf.name_scope("get_similarity_transform_for_cv2"):
        trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
        cv2_trans = cvt_tform_mat_for_cv2(trans)

    return cv2_trans
