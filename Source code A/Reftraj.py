import numpy as np
import os

if not os.path.exists("Reference_traj_circle"):
    os.makedirs("Reference_traj_circle")

    
# quadrotor 0
# coeffx1  = np.array([[1,	0,	0,	0,	0.0159994512651706,	-0.00312900383912878,	0.000190805145299291,	-3.35519827605777e-06]])
# coeffx2  = np.array([[6,	1.16010809176743,	-0.245710324268002,	-0.0676444803806955,	0.00297984089720244,	0.00133549715716280,	3.81436238190747e-05,	-1.65660770008684e-05]])
# coeffx3  = np.array([[6,	-1.30677894577231,	-0.371207194759780,	0.0619410830859453,	0.0125067447044141,	-0.00110890616717792,	-0.000309743993290792,	3.24707797208004e-05]])
# coeffx4  = np.array([[1,	-1.24589979385470,	0.351581539072783,	0.0370133686961929,	-0.0152574000613400,	-0.000547320679256936,	0.000372142380759335,	-2.32576091298607e-05]])

# coeffy1  = np.array([[0,	0,	0,	0,	0.0207493201314231,	-0.00666908154259950,	0.000686078835810112,	-2.32576091670590e-05]])
# coeffy2  = np.array([[0,	-1.24589979301679,	-0.351581539378058,	0.0370133686246630,	0.0152574000789763,	-0.000547320679402213,	-0.000372142381378565,	3.24707797690752e-05]])
# coeffy3  = np.array([[-5,	-1.30677894602402,	0.371207194844571,	0.0619410831285406,	-0.0125067447163621,	-0.00110890616794306,	0.000309743993681183,	-1.65660770381382e-05]])
# coeffy4  = np.array([[-5,	1.16010809177182,	0.245710324154955,	-0.0676444803994014,	-0.00297984088961051,	0.00133549715805995,	-3.81436240386627e-05,	-3.35519825905335e-06]])

# coeffz1  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz2  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz3  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz4  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])

# np.save('Reference_traj_circle/coeffx0_1',coeffx1)
# np.save('Reference_traj_circle/coeffx0_2',coeffx2)
# np.save('Reference_traj_circle/coeffx0_3',coeffx3)
# np.save('Reference_traj_circle/coeffx0_4',coeffx4)
# np.save('Reference_traj_circle/coeffy0_1',coeffy1)
# np.save('Reference_traj_circle/coeffy0_2',coeffy2)
# np.save('Reference_traj_circle/coeffy0_3',coeffy3)
# np.save('Reference_traj_circle/coeffy0_4',coeffy4)
# np.save('Reference_traj_circle/coeffz0_1',coeffz1)
# np.save('Reference_traj_circle/coeffz0_2',coeffz2)
# np.save('Reference_traj_circle/coeffz0_3',coeffz3)
# np.save('Reference_traj_circle/coeffz0_4',coeffz4)

# # quadrotor 1
# coeffx1  = np.array([[0,	0,	0,	0,	0.0159994512651706,	-0.00312900383912878,	0.000190805145299291,	-3.35519827605777e-06]])
# coeffx2  = np.array([[5,	1.16010809176743,	-0.245710324268002,	-0.0676444803806955,	0.00297984089720244,	0.00133549715716280,	3.81436238190747e-05,	-1.65660770008684e-05]])
# coeffx3  = np.array([[5,	-1.30677894577231,	-0.371207194759780,	0.0619410830859453,	0.0125067447044141,	-0.00110890616717792,	-0.000309743993290792,	3.24707797208004e-05]])
# coeffx4  = np.array([[0,	-1.24589979385470,	0.351581539072783,	0.0370133686961929,	-0.0152574000613400,	-0.000547320679256936,	0.000372142380759335,	-2.32576091298607e-05]])

# coeffy1  = np.array([[1,	0,	0,	0,	0.0207493201314231,	-0.00666908154259950,	0.000686078835810112,	-2.32576091670590e-05]])
# coeffy2  = np.array([[1,	-1.24589979301679,	-0.351581539378058,	0.0370133686246630,	0.0152574000789763,	-0.000547320679402213,	-0.000372142381378565,	3.24707797690752e-05]])
# coeffy3  = np.array([[-4,	-1.30677894602402,	0.371207194844571,	0.0619410831285406,	-0.0125067447163621,	-0.00110890616794306,	0.000309743993681183,	-1.65660770381382e-05]])
# coeffy4  = np.array([[-4,	1.16010809177182,	0.245710324154955,	-0.0676444803994014,	-0.00297984088961051,	0.00133549715805995,	-3.81436240386627e-05,	-3.35519825905335e-06]])

# coeffz1  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz2  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz3  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz4  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])

# np.save('Reference_traj_circle/coeffx1_1',coeffx1)
# np.save('Reference_traj_circle/coeffx1_2',coeffx2)
# np.save('Reference_traj_circle/coeffx1_3',coeffx3)
# np.save('Reference_traj_circle/coeffx1_4',coeffx4)
# np.save('Reference_traj_circle/coeffy1_1',coeffy1)
# np.save('Reference_traj_circle/coeffy1_2',coeffy2)
# np.save('Reference_traj_circle/coeffy1_3',coeffy3)
# np.save('Reference_traj_circle/coeffy1_4',coeffy4)
# np.save('Reference_traj_circle/coeffz1_1',coeffz1)
# np.save('Reference_traj_circle/coeffz1_2',coeffz2)
# np.save('Reference_traj_circle/coeffz1_3',coeffz3)
# np.save('Reference_traj_circle/coeffz1_4',coeffz4)

# # quadrotor 2
# coeffx1  = np.array([[-1,	0,	0,	0,	0.0159994512651706,	-0.00312900383912878,	0.000190805145299291,	-3.35519827605777e-06]])
# coeffx2  = np.array([[4,	1.16010809176743,	-0.245710324268002,	-0.0676444803806955,	0.00297984089720244,	0.00133549715716280,	3.81436238190747e-05,	-1.65660770008684e-05]])
# coeffx3  = np.array([[4,	-1.30677894577231,	-0.371207194759780,	0.0619410830859453,	0.0125067447044141,	-0.00110890616717792,	-0.000309743993290792,	3.24707797208004e-05]])
# coeffx4  = np.array([[-1,	-1.24589979385470,	0.351581539072783,	0.0370133686961929,	-0.0152574000613400,	-0.000547320679256936,	0.000372142380759335,	-2.32576091298607e-05]])

# coeffy1  = np.array([[0,	0,	0,	0,	0.0207493201314231,	-0.00666908154259950,	0.000686078835810112,	-2.32576091670590e-05]])
# coeffy2  = np.array([[0,	-1.24589979301679,	-0.351581539378058,	0.0370133686246630,	0.0152574000789763,	-0.000547320679402213,	-0.000372142381378565,	3.24707797690752e-05]])
# coeffy3  = np.array([[-5,	-1.30677894602402,	0.371207194844571,	0.0619410831285406,	-0.0125067447163621,	-0.00110890616794306,	0.000309743993681183,	-1.65660770381382e-05]])
# coeffy4  = np.array([[-5,	1.16010809177182,	0.245710324154955,	-0.0676444803994014,	-0.00297984088961051,	0.00133549715805995,	-3.81436240386627e-05,	-3.35519825905335e-06]])

# coeffz1  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz2  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz3  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz4  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])

# np.save('Reference_traj_circle/coeffx2_1',coeffx1)
# np.save('Reference_traj_circle/coeffx2_2',coeffx2)
# np.save('Reference_traj_circle/coeffx2_3',coeffx3)
# np.save('Reference_traj_circle/coeffx2_4',coeffx4)
# np.save('Reference_traj_circle/coeffy2_1',coeffy1)
# np.save('Reference_traj_circle/coeffy2_2',coeffy2)
# np.save('Reference_traj_circle/coeffy2_3',coeffy3)
# np.save('Reference_traj_circle/coeffy2_4',coeffy4)
# np.save('Reference_traj_circle/coeffz2_1',coeffz1)
# np.save('Reference_traj_circle/coeffz2_2',coeffz2)
# np.save('Reference_traj_circle/coeffz2_3',coeffz3)
# np.save('Reference_traj_circle/coeffz2_4',coeffz4)

# # quadrotor 3
# coeffx1  = np.array([[0,	0,	0,	0,	0.0159994512651706,	-0.00312900383912878,	0.000190805145299291,	-3.35519827605777e-06]])
# coeffx2  = np.array([[5,	1.16010809176743,	-0.245710324268002,	-0.0676444803806955,	0.00297984089720244,	0.00133549715716280,	3.81436238190747e-05,	-1.65660770008684e-05]])
# coeffx3  = np.array([[5,	-1.30677894577231,	-0.371207194759780,	0.0619410830859453,	0.0125067447044141,	-0.00110890616717792,	-0.000309743993290792,	3.24707797208004e-05]])
# coeffx4  = np.array([[0,	-1.24589979385470,	0.351581539072783,	0.0370133686961929,	-0.0152574000613400,	-0.000547320679256936,	0.000372142380759335,	-2.32576091298607e-05]])

# coeffy1  = np.array([[-1,	0,	0,	0,	0.0207493201314231,	-0.00666908154259950,	0.000686078835810112,	-2.32576091670590e-05]])
# coeffy2  = np.array([[-1,	-1.24589979301679,	-0.351581539378058,	0.0370133686246630,	0.0152574000789763,	-0.000547320679402213,	-0.000372142381378565,	3.24707797690752e-05]])
# coeffy3  = np.array([[-6,	-1.30677894602402,	0.371207194844571,	0.0619410831285406,	-0.0125067447163621,	-0.00110890616794306,	0.000309743993681183,	-1.65660770381382e-05]])
# coeffy4  = np.array([[-6,	1.16010809177182,	0.245710324154955,	-0.0676444803994014,	-0.00297984088961051,	0.00133549715805995,	-3.81436240386627e-05,	-3.35519825905335e-06]])

# coeffz1  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz2  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz3  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz4  = np.array([[1.5000,         0,         0,         0,         0,         0,         0,         0]])

# np.save('Reference_traj_circle/coeffx3_1',coeffx1)
# np.save('Reference_traj_circle/coeffx3_2',coeffx2)
# np.save('Reference_traj_circle/coeffx3_3',coeffx3)
# np.save('Reference_traj_circle/coeffx3_4',coeffx4)
# np.save('Reference_traj_circle/coeffy3_1',coeffy1)
# np.save('Reference_traj_circle/coeffy3_2',coeffy2)
# np.save('Reference_traj_circle/coeffy3_3',coeffy3)
# np.save('Reference_traj_circle/coeffy3_4',coeffy4)
# np.save('Reference_traj_circle/coeffz3_1',coeffz1)
# np.save('Reference_traj_circle/coeffz3_2',coeffz2)
# np.save('Reference_traj_circle/coeffz3_3',coeffz3)
# np.save('Reference_traj_circle/coeffz3_4',coeffz4)

# # payload
# coeffx1  = np.array([[0,	0,	0,	0,	0.0159994512651706,	-0.00312900383912878,	0.000190805145299291,	-3.35519827605777e-06]])
# coeffx2  = np.array([[5,	1.16010809176743,	-0.245710324268002,	-0.0676444803806955,	0.00297984089720244,	0.00133549715716280,	3.81436238190747e-05,	-1.65660770008684e-05]])
# coeffx3  = np.array([[5,	-1.30677894577231,	-0.371207194759780,	0.0619410830859453,	0.0125067447044141,	-0.00110890616717792,	-0.000309743993290792,	3.24707797208004e-05]])
# coeffx4  = np.array([[0,	-1.24589979385470,	0.351581539072783,	0.0370133686961929,	-0.0152574000613400,	-0.000547320679256936,	0.000372142380759335,	-2.32576091298607e-05]])

# coeffy1  = np.array([[0,	0,	0,	0,	0.0207493201314231,	-0.00666908154259950,	0.000686078835810112,	-2.32576091670590e-05]])
# coeffy2  = np.array([[0,	-1.24589979301679,	-0.351581539378058,	0.0370133686246630,	0.0152574000789763,	-0.000547320679402213,	-0.000372142381378565,	3.24707797690752e-05]])
# coeffy3  = np.array([[-5,	-1.30677894602402,	0.371207194844571,	0.0619410831285406,	-0.0125067447163621,	-0.00110890616794306,	0.000309743993681183,	-1.65660770381382e-05]])
# coeffy4  = np.array([[-5,	1.16010809177182,	0.245710324154955,	-0.0676444803994014,	-0.00297984088961051,	0.00133549715805995,	-3.81436240386627e-05,	-3.35519825905335e-06]])

# coeffz1  = np.array([[0.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz2  = np.array([[0.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz3  = np.array([[0.5000,         0,         0,         0,         0,         0,         0,         0]])
# coeffz4  = np.array([[0.5000,         0,         0,         0,         0,         0,         0,         0]])

# # new payload: angle trajectory, 10 s
# # coeffa   = np.array([[0,	0,	0,	0,	0.0219911485751286,	-0.00527787565803085,	0.000439822971502571,	-1.25663706143592e-05]])
# # coeffa = np.array([[0,	0,	0,	0,	0.00137444678594555,	-0.000164933614313466,	6.87223392972776e-06,	-9.81747704246823e-08]])
# coeffa = np.array([[0,	0,	0,	0,	0.00434393058274145,	-0.000695028893238633,	3.86127162910351e-05,	-7.35480310305431e-07]]) # 15s for training and testing in the 1st simulation
# coeffa = np.array([[0,	0,	0,	0,	0.00209487393072021,	-0.000279316524096028,	1.29313205600013e-05,	-2.05259056507957e-07]]) # 18s
coeffa = np.array([[0,	0,	0,	0,	0.00137444678594555,	-0.000164933614313466,	6.87223392972776e-06,	-9.81747704246823e-08]]) # 20s for training and testing in the 2nd simulation
# coeffa = np.array([[0,	0,	0,	0,	0.000271495661421339,	-2.17196529137071e-05,	6.03323692047418e-07,	-5.74593992426112e-09]]) # 30s
# np.save('Reference_traj_circle/coeffxl_1',coeffx1)
# np.save('Reference_traj_circle/coeffxl_2',coeffx2)
# np.save('Reference_traj_circle/coeffxl_3',coeffx3)
# np.save('Reference_traj_circle/coeffxl_4',coeffx4)
# np.save('Reference_traj_circle/coeffyl_1',coeffy1)
# np.save('Reference_traj_circle/coeffyl_2',coeffy2)
# np.save('Reference_traj_circle/coeffyl_3',coeffy3)
# np.save('Reference_traj_circle/coeffyl_4',coeffy4)
# np.save('Reference_traj_circle/coeffzl_1',coeffz1)
# np.save('Reference_traj_circle/coeffzl_2',coeffz2)
# np.save('Reference_traj_circle/coeffzl_3',coeffz3)
# np.save('Reference_traj_circle/coeffzl_4',coeffz4)
np.save('Reference_traj_circle/coeffa',coeffa)


# new payload minisnap trajectory, time = 15.8
# coeffx1 = np.array([[0,	0,	0,	0,	-0.00569384708443376,	0.00524455346637073,	-0.000886898899427708,	4.29518719629043e-05]])
# coeffx2 = np.array([[3,	1.73199988130206,	0.0916811094782298,	-0.108490668093207,	-0.0153024072652927,	0.00281103744009310,	0.000736681860892091,	-0.000105609640789732]])
# coeffx3 = np.array([[6,	-1.18744116761591e-10,	-0.641547316881916,	1.48159386981189e-11,	0.0311442128876047,	-3.02467615811296e-14,	-0.00111148685292983,	0.000105609640789826]])
# coeffx4 = np.array([[3,	-1.73199988075592,	0.0916811097674330,	0.108490668078459,	-0.0153024072679752,	-0.00281103744033707,	0.000736681860769491,	-4.29518719562680e-05]])

# coeffy1 = np.array([[0,	0,	0,	0,	0.0461312504597524,	-0.0147943913425831,	0.00159271036531923,	-5.81518834088372e-05]])
# coeffy2 = np.array([[3,	-0.0568060950833151,	-0.517633958734110,	-0.0323591302339987,	0.0228452120824883,	0.00119953716963643,	-0.000605430827541181,	3.45960473175561e-05]])
# coeffy3 = np.array([[0,	-1.88521974334866,	4.09994989070083e-11,	0.129166213520013,	-2.21255615723514e-12,	-0.00334119403305491,	3.83514291940488e-13,	3.45960472083100e-05]])
# coeffy4 = np.array([[-3,	-0.0568060949105225,	0.517633958862144,	-0.0323591303010103,	-0.0228452120624079,	0.00119953716867972,	0.000605430826524852,	-5.81518833064674e-05]])

# coeffz1 = np.array([[5,	0,	0,	0,	-0.00189794902814460,	0.00174818448879025,	-0.000295632966475905,	1.43172906543016e-05]])
# coeffz2 = np.array([[6,	0.577333293767196,	0.0305603698263891,	-0.0361635560298186,	-0.00510080242176428,	0.000937012480031087,	0.000245560620297342,	-3.52032135965748e-05]])
# coeffz3 = np.array([[7,	-3.94240804446567e-11,	-0.213849105627924,	7.42572403566928e-12,	0.0103814042958683,	-1.01598539349033e-14,	-0.000370495617643246,	3.52032135966053e-05]])
# coeffz4 = np.array([[6,	-0.577333293585312,	0.0305603699280703,	0.0361635560409975,	-0.00510080242265701,	-0.000937012480112873,	0.000245560620256570,	-1.43172906520930e-05]])

# new tilted circle minisnap, time =14.8
# coeffx1 = np.array([[3,	0,	0,	0,	0.00338780038909970,	-0.00571172273081262,	0.00111055674470965,	-5.91448428222220e-05]])
# coeffx2 = np.array([[0,	-1.80126155961536,	-0.101394183698723,	0.122423750117153,	0.0182948240377098,	-0.00344606287116043,	-0.000959512754195524,	0.000142716985236779]])
# coeffx3 = np.array([[-3,	9.09132786830362e-11,	0.697071604616694,	-5.81842698465996e-11,	-0.0369076462417281,	2.50261967431026e-12,	0.00143813259778400,	-0.000142716985304754]])
# coeffx4 = np.array([[0,	1.80126155911532,	-0.101394183807446,	-0.122423750089070,	0.0182948240669483,	0.00344606286815687,	-0.000959512755208169,	5.91448429619241e-05]])
# coeffy1 = np.array([[0,	0,	0,	0,	0.0603167293159471,	-0.0207119597329688,	0.00238983212872571,	-9.36218709539654e-05]])
# coeffy2 = np.array([[3,	0.00295077046384334,	-0.581699343370030,	-0.0450534522252476,	0.0291090988402321,	0.00183152187809900,	-0.000886933354666368,	5.27936518559574e-05]])
# coeffy3 = np.array([[0,	-2.00740134247241,	-2.01128926846627e-10,	0.155975056155531,	4.82808766054133e-11,	-0.00455439830060397,	-3.62791109226313e-12,	5.27936526719228e-05]])
# coeffy4 = np.array([[-3,	0.00295077157680778,	0.581699344159643,	-0.0450534520126776,	-0.0291090989636439,	0.00183152187422608,	0.000886933361260144,	-9.36218715626703e-05]])
# coeffz1 = np.array([[5,	0,	0,	0,	-0.00112926679636639,	0.00190390757693745,	-0.000370185581569867,	1.97149476074065e-05]])
# coeffz2 = np.array([[6,	0.600420519871318,	0.0337980612342290,	-0.0408079167007521,	-0.00609827467923710,	0.00114868762372053,	0.000319837584731714,	-4.75723284122452e-05]])
# coeffz3 = np.array([[7,	-3.15426387231773e-11,	-0.232357201538888,	1.62666751180022e-11,	0.0123025487472425,	-8.34051626532673e-13,	-0.000479377532594711,	4.75723284349228e-05]])
# coeffz4 = np.array([[6,	-0.600420519705344,	0.0337980612654422,	0.0408079167009449,	-0.00609827468898177,	-0.00114868762271939,	0.000319837585069458,	-1.97149476539785e-05]])


# np.save('Reference_traj_circle/coeffxl_1',coeffx1)
# np.save('Reference_traj_circle/coeffxl_2',coeffx2)
# np.save('Reference_traj_circle/coeffxl_3',coeffx3)
# np.save('Reference_traj_circle/coeffxl_4',coeffx4)
# np.save('Reference_traj_circle/coeffyl_1',coeffy1)
# np.save('Reference_traj_circle/coeffyl_2',coeffy2)
# np.save('Reference_traj_circle/coeffyl_3',coeffy3)
# np.save('Reference_traj_circle/coeffyl_4',coeffy4)
# np.save('Reference_traj_circle/coeffzl_1',coeffz1)
# np.save('Reference_traj_circle/coeffzl_2',coeffz2)
# np.save('Reference_traj_circle/coeffzl_3',coeffz3)
# np.save('Reference_traj_circle/coeffzl_4',coeffz4)

# test figure-8 trajectory, 20s
# coeffx1 = np.array([[0,	0,	0,	0,	-0.000224297400166642,	0.00862420804239762,	-0.00227335683319544,	0.000154901165026614]])
# coeffx2 = np.array([[2,	1.45538798031746,	0.0992650099779265,	-0.145697779517872,	-0.0263671668595364,	0.00611043549461177,	0.00206387578765621,	-0.000375008447298643]])
# coeffx3 = np.array([[4,	-0.0225794665368828,	-0.675574319487593,	0.00799770101023531,	0.0535673701023049,	-0.000623764626422710,	-0.00318624247453155,	0.000392786468355779]])
# coeffx4 = np.array([[2,	-1.40043987681966,	0.107381911731694,	0.121747703104213,	-0.0338646134941441,	-0.00586461097896644,	0.00231276808234965,	-0.000165197720194843]])
# coeffx5 = np.array([[0,	-0.692733409430608,	-6.14437304819785e-11,	-0.106221474111242,	2.58607054435393e-12,	0.00801199751286455,	-4.21410059663396e-13,	-0.000165197720071568]])
# coeffx6 = np.array([[-2,	-1.40043987675138,	-0.107381911621810,	0.121747703092933,	0.0338646134859252,	-0.00586461097815736,	-0.00231276808152646,	0.000392786468215311]])
# coeffx7 = np.array([[-4,	-0.0225794669065323,	0.675574319512692,	0.00799770098058194,	-0.0535673700869177,	-0.000623764626567544,	0.00318624247348969,	-0.000375008447177261]])
# coeffx8 = np.array([[-2,	1.45538798080495,	-0.0992650096722002,	-0.145697779503054,	0.0263671668471065,	0.00611043549245515,	-0.00206387578688476,	0.000154901164996078]])

# coeffy1 = np.array([[0,	0,	0,	0,	0.0976038143684468,	-0.0420466530642218,	0.00610887399409180,	-0.000302291969016065]])
# coeffy2 = np.array([[2,	0.0324670401902826,	-0.582302146139826,	-0.0549807903165139,	0.0456665010699920,	0.00299622120471753,	-0.00235530113836564,	0.000221226594203875]])
# coeffy3 = np.array([[0,	-1.60859360204795,	0.00689883218376806,	0.177238777044445,	-0.00374590880763266,	-0.00668435854255564,	0.000741871180370736,	-1.53443928524593e-05]])
# coeffy4 = np.array([[-2,	0.0268142646078160,	0.613418650959734,	-0.00999630625410200,	-0.0303736534096165,	0.000929166622141250,	0.000527049680442973,	-3.76464058241291e-05]])
# coeffy5 = np.array([[0,	1.54723756296184,	-4.32073672367011e-11,	-0.152572907026709,	7.57298971350957e-12,	0.00409146469823586,	-9.82426730712235e-13,	-3.76464054968413e-05]])
# coeffy6 = np.array([[2,	0.0268142647025564,	-0.613418650910228,	-0.00999630628251026,	0.0303736533918475,	0.000929166624856968,	-0.000527049677940728,	-1.53443933309000e-05]])
# coeffy7 = np.array([[0,	-1.60859360210561,	-0.00689883216908346,	0.177238777132602,	0.00374590883137237,	-0.00668435855022875,	-0.000741871184690535,	0.000221226595005959]])
# coeffy8 = np.array([[-2,	0.0324670402307242,	0.582302145980588,	-0.0549807902921848,	-0.0456665011506847,	0.00299622121385544,	0.00235530114539465,	-0.000302291970034600]])

# test figure-8 trajectory, 36s
# coeffx1 = np.array([[3,	0,	0,	0,	-0.00294308266226649,	0.000581963741808567,	-4.40756190972900e-05,	1.23341463262000e-06]])
# coeffx2 = np.array([[2,	-0.425267106185264,	-0.0340825321460627,	0.00441397639200676,	3.96099020628609e-05,	-7.22970834259340e-05,	7.72779544323153e-06,	-2.35993752431284e-07]])
# coeffx3 = np.array([[0,	-0.527743466064823,	0.00101752661441513,	0.00125727562238993,	-8.02868655258613e-05,	3.38761063944817e-05,	1.11997033851076e-06,	-2.74371050881489e-07]])
# coeffx4 = np.array([[-2,	-0.437431713669733,	0.0284786141065161,	0.00436806021452537,	0.000251436989633261,	-3.14332785833304e-05,	-6.56241911655769e-06,	5.62295646858699e-07]])
# coeffx5 = np.array([[-3,	-2.01518061636425e-11,	0.0718079055755829,	1.83017241433334e-12,	-0.000692666921043489,	-5.12639842740560e-14,	9.18185899550264e-06,	-5.62295646297474e-07]])
# coeffx6 = np.array([[-2,	0.437431713728246,	0.0284786141117245,	-0.00436806021927683,	0.000251436989143984,	3.14332786790021e-05,	-6.56241907045428e-06,	2.74371049045615e-07]])
# coeffx7 = np.array([[0,	0.527743465979253,	0.00101752660032976,	-0.00125727561507630,	-8.02868643232665e-05,	-3.38761065328263e-05,	1.11997033949060e-06,	2.35993755908941e-07]])
# coeffx8 = np.array([[2,	0.425267106362654,	-0.0340825321013806,	-0.00441397639697746,	3.96098997324411e-05,	7.22970836075017e-05,	7.72779553442673e-06,	-1.23341463843419e-06]])

# coeffy1 = np.array([[0,	0,	0,	0,	0.0226527678149821,	-0.00616401623899525,	0.000571870229873802,	-1.82462214891329e-05]])
# coeffy2 = np.array([[3,	0.351273639868597,	-0.283654943736028,	-0.0325486321688463,	0.00860077031911885,	0.000629168590737122,	-0.000194471072645582,	1.01919356766127e-05]])
# coeffy3 = np.array([[0,	-1.37577434687987,	0.0265017847471307,	0.0481274381305473,	-0.00265897938544927,	-0.000613646765421228,	9.09031262039448e-05,	-3.92782001774147e-06]])
# coeffy4 = np.array([[-3,	0.126080730725243,	0.220639254397471,	-0.0114369803224921,	-0.00191348124467117,	0.000248280737466196,	-1.90758342840187e-05,	6.81279789232213e-07]])
# coeffy5 = np.array([[0,	1.07249958403868,	-4.56577330232373e-11,	-0.0206405632150372,	4.36143424235049e-12,	1.93707238318527e-05,	-1.10936803207336e-13,	6.81279812448242e-07]])
# coeffy6 = np.array([[3,	0.126080731075608,	-0.220639254281544,	-0.0114369803586742,	0.00191348123425320,	0.000248280738198249,	1.90758346462118e-05,	-3.92782004942661e-06]])
# coeffy7 = np.array([[0,	-1.37577434777278,	-0.0265017849576948,	0.0481274382092115,	0.00265897940260880,	-0.000613646766894343,	-9.09031268331244e-05,	1.01919357256891e-05]])
# coeffy8 = np.array([[-3,	0.351273642083667,	0.283654944291561,	-0.0325486322741640,	-0.00860077034969421,	0.000629168592883064,	0.000194471073509911,	-1.82462215678138e-05]])

# coeffx1 = np.array([[0,	0,	0,	0,	0.0452754781639629,	-0.0116172822235300,	0.000848805156856451,	-1.04450278054073e-05]])
# coeffx2 = np.array([[3,	1.63598021034416,	-0.0538127949180818,	-0.141474353508460,	-0.00675379094524848,	0.00524451219840959,	0.000556344378419849,	-0.000158470222120648]])
# coeffx3 = np.array([[5,	-0.0377000845850101,	-0.618158261178135,	0.0145475830408805,	0.0347003315502391,	-0.00139085391853537,	-0.00166223873127662,	0.000213576945073963]])
# coeffx4 = np.array([[3,	-1.56008720171367,	-0.0647467075650313,	0.0901609709392259,	-0.0191409868909665,	-0.00339725530764574,	0.00132783849964805,	-9.48456071659029e-05]])
# coeffx5 = np.array([[0,	-1.40898022580286,	-4.44580869809661e-11,	-0.0395165165540577,	4.99724665668267e-12,	0.00456977568619513,	-7.69648839098113e-13,	-9.48456069391195e-05]])
# coeffx6 = np.array([[-3,	-1.56008720163758,	0.0647467076464665,	0.0901609709317970,	0.0191409868778339,	-0.00339725530592607,	-0.00132783849802872,	0.000213576944767145]])
# coeffx7 = np.array([[-5,	-0.0377000848678725,	0.618158261168499,	0.0145475830887673,	-0.0347003315283249,	-0.00139085392198371,	0.00166223872871166,	-0.000158470221688694]])
# coeffx8 = np.array([[-3,	1.63598021074336,	0.0538127950857939,	-0.141474353582023,	0.00675379090165773,	0.00524451220070476,	-0.000556344374812206,	-1.04450282616360e-05]])

# coeffy1 = np.array([[0,	0,	0,	0,	0.195207628736894,	-0.0840933061284437,	0.0122177479881836,	-0.000604583938032130]])
# coeffy2 = np.array([[4,	0.0649340803805651,	-1.16460429227965,	-0.109961580633028,	0.0913330021399840,	0.00599244240943506,	-0.00471060227673128,	0.000442453188407751]])
# coeffy3 = np.array([[0,	-3.21718720409591,	0.0137976643675361,	0.354477554088891,	-0.00749181761526532,	-0.0133687170851113,	0.00148374236074147,	-3.06887857049186e-05]])
# coeffy4 = np.array([[-4,	0.0536285292156321,	1.22683730191947,	-0.0199926125082040,	-0.0607473068192329,	0.00185833324428250,	0.00105409936088595,	-7.52928116482582e-05]])
# coeffy5 = np.array([[0,	3.09447512592369,	-8.64147344734022e-11,	-0.305145814053418,	1.51459794270191e-11,	0.00818292939647172,	-1.96485346142447e-12,	-7.52928109936826e-05]])
# coeffy6 = np.array([[4,	0.0536285294051128,	-1.22683730182046,	-0.0199926125650205,	0.0607473067836951,	0.00185833324971394,	-0.00105409935588146,	-3.06887866618000e-05]])
# coeffy7 = np.array([[0,	-3.21718720421122,	-0.0137976643381669,	0.354477554265205,	0.00749181766274475,	-0.0133687171004575,	-0.00148374236938107,	0.000442453190011917]])
# coeffy8 = np.array([[-4,	0.0649340804614483,	1.16460429196118,	-0.109961580584370,	-0.0913330023013694,	0.00599244242771088,	0.00471060229078930,	-0.000604583940069201]])


# coeffz1 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz2 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz3 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz4 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz5 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz6 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz7 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz8 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])

# little faster

coeffx1 = np.array([[0,	0,	0,	0,	0.0680253659460270,	-0.0217380273564934,	0.00240988615188231,	-9.31181242214113e-05]])
coeffx2 = np.array([[3.50000000000000,	1.72627632536031,	-0.130351697367021,	-0.139362640518661,	0.00305289701189589,	0.00481155055030830,	-0.000197421326198267,	-5.02011095316597e-05]])
coeffx3 = np.array([[5.50000000000000,	-0.0452603936090772,	-0.589450232025734,	0.0178225240710956,	0.0252668122742068,	-0.00177439856459259,	-0.000900236859648805,	0.000123972183433008]])
coeffx4 = np.array([[3.50000000000000,	-1.63991086416113,	-0.150811017214343,	0.0743676048622925,	-0.0117791735893771,	-0.00216357747198605,	0.000835373708297492,	-5.96695506514668e-05]])
coeffx5 = np.array([[0,	-1.76710363398852,	-3.40774836792999e-11,	-0.00616403776802375,	6.20198018317085e-12,	0.00284866477286124,	-9.44078504907391e-13,	-5.96695503728528e-05]])
coeffx6 = np.array([[-3.50000000000000,	-1.63991086408069,	0.150811017281830,	0.0743676048382186,	0.0117791735737886,	-0.00216357746981077,	-0.000835373706279737,	0.000123972183043046]])
coeffx7 = np.array([[-5.50000000000000,	-0.0452603938480910,	0.589450231999206,	0.0178225241281472,	-0.0252668122490300,	-0.00177439856969039,	0.000900236856322125,	-5.02011089443412e-05]])
coeffx8 = np.array([[-3.50000000000000,	1.72627632571089,	0.130351697474732,	-0.139362640688166,	-0.00305289707106339,	0.00481155055482790,	0.000197421331224382,	-9.31181248905141e-05]])

coeffy1 = np.array([[0,	0,	0,	0,	0.219608582329019,	-0.0946049693945083,	0.0137449664867086,	-0.000680156930286298]])
coeffy2 = np.array([[4.50000000000000,	0.0730508404299590,	-1.31017982880993,	-0.123706778214945,	0.102749627407481,	0.00674149771061185,	-0.00529942756132145,	0.000497759836958551]])
coeffy3 = np.array([[0,	-3.61933560460838,	0.0155223724097426,	0.398787248364837,	-0.00842829481717341,	-0.0150398067207480,	0.00166921015583321,	-3.45248839179082e-05]])
coeffy4 = np.array([[-4.50000000000000,	0.0603320953667110,	1.38019196465838,	-0.0224916890686879,	-0.0683407201716371,	0.00209062489981667,	0.00118586178099718,	-8.47044131043555e-05]])
coeffy5 = np.array([[0,	3.48128451666530,	-9.56247939612098e-11,	-0.343289040820439,	1.70392383541249e-11,	0.00920579557103124,	-2.21070008083291e-12,	-8.47044123678618e-05]])
coeffy6 = np.array([[4.50000000000000,	0.0603320955812575,	-1.38019196454931,	-0.0224916891383665,	0.0683407201316568,	0.00209062490592802,	-0.00118586177536654,	-3.45248849945383e-05]])
coeffy7 = np.array([[0,	-3.61933560474181,	-0.0155223724063336,	0.398787248503369,	0.00842829487058663,	-0.0150398067380129,	-0.00166921016555446,	0.000497759838763511]])
coeffy8 = np.array([[-4.50000000000000,	0.0730508405319780,	1.31017982848072,	-0.123706778700484,	-0.102749627589044,	0.00674149773117671,	0.00529942757713755,	-0.000680156932577821]])

coeffz1 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
coeffz2 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
coeffz3 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
coeffz4 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
coeffz5 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
coeffz6 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
coeffz7 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
coeffz8 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])

# faster

# coeffx1 = np.array([[0,	0,	0,	0,	0.0907752537280929,	-0.0318587724894579,	0.00397096714690840,	-0.000175791220637433]])
# coeffx2 = np.array([[4,	1.81657244037506,	-0.206890599815025,	-0.137250927528863,	0.0128595849690398,	0.00437858890220712,	-0.000951187030816427,	5.80680030573374e-05]])
# coeffx3 = np.array([[6,	-0.0528207026317434,	-0.560742202867748,	0.0210974650919817,	0.0158332929981742,	-0.00215794321064899,	-0.000138234988021333,	3.43674217920995e-05]])
# coeffx4 = np.array([[4,	-1.71973452660883,	-0.236875326863162,	0.0585742387742221,	-0.00441736028778823,	-0.000929899636325873,	0.000342908916946772,	-2.44934941370088e-05]])
# coeffx5 = np.array([[0,	-2.12522704217396,	-2.56083036460456e-11,	0.0271884410180162,	7.40730349084464e-12,	0.00112755385952690,	-1.11835166122969e-12,	-2.44934938066070e-05]])
# coeffx6 = np.array([[-4,	-1.71973452652520,	0.236875326918694,	0.0585742387437762,	0.00441736026974221,	-0.000929899633694388,	-0.000342908914531154,	3.43674213190019e-05]])
# coeffx7 = np.array([[-6,	-0.0528207028301864,	0.560742202829082,	0.0210974651712494,	-0.0158332929697344,	-0.00215794321739760,	0.000138234983932780,	5.80680037999861e-05]])
# coeffx8 = np.array([[-4,	1.81657244068265,	0.206890599838257,	-0.137250927573973,	-0.0128595850437894,	0.00437858890895346,	0.000951187037260518,	-0.000175791221519362]])

# coeffy1 = np.array([[0,	0,	0,	0,	0.244009535921117,	-0.105116632660555,	0.0152721849852295,	-0.000755729922540163]])
# coeffy2 = np.array([[5,	0.0811676004794274,	-1.45575536535144,	-0.137451975789412,	0.114166252674983,	0.00749055301179057,	-0.00588825284591285,	0.000553066485509517]])
# coeffy3 = np.array([[0,	-4.02148400512040,	0.0172470804594068,	0.443096942618606,	-0.00936477201908305,	-0.0167108963563876,	0.00185467795092624,	-3.83609821310622e-05]])
# coeffy4 = np.array([[-5,	0.0670356615191799,	1.53354662739815,	-0.0249907656310704,	-0.0759341335240404,	0.00232291655535237,	0.00131762420110773,	-9.41160145603658e-05]])
# coeffy5 = np.array([[0,	3.86809390740507,	-1.10335323902319e-10,	-0.381432267579717,	1.89321100375861e-11,	0.0102286617455899,	-2.45617193433609e-12,	-9.41160137420880e-05]])
# coeffy6 = np.array([[5,	0.0670356617572545,	-1.53354662727460,	-0.0249907657197672,	0.0759341334796194,	0.00232291656214186,	-0.00131762419485157,	-3.83609833272853e-05]])
# coeffy7 = np.array([[0,	-4.02148400526867,	-0.0172470804441507,	0.443096942801820,	0.00936477207842922,	-0.0167108963755698,	-0.00185467796172724,	0.000553066487515023]])
# coeffy8 = np.array([[-5,	0.0811676005878158,	1.45575536495422,	-0.137451976566697,	-0.114166252876712,	0.00749055303463936,	0.00588825286348645,	-0.000755729925086488]])

# coeffz1 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz2 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz3 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz4 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz5 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz6 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz7 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])
# coeffz8 = np.array([[2,	0,	0,	0,	0,	0,	0,	0]])


np.save('Reference_traj_fig8/coeffxl_1',coeffx1)
np.save('Reference_traj_fig8/coeffxl_2',coeffx2)
np.save('Reference_traj_fig8/coeffxl_3',coeffx3)
np.save('Reference_traj_fig8/coeffxl_4',coeffx4)
np.save('Reference_traj_fig8/coeffxl_5',coeffx5)
np.save('Reference_traj_fig8/coeffxl_6',coeffx6)
np.save('Reference_traj_fig8/coeffxl_7',coeffx7)
np.save('Reference_traj_fig8/coeffxl_8',coeffx8)

np.save('Reference_traj_fig8/coeffyl_1',coeffy1)
np.save('Reference_traj_fig8/coeffyl_2',coeffy2)
np.save('Reference_traj_fig8/coeffyl_3',coeffy3)
np.save('Reference_traj_fig8/coeffyl_4',coeffy4)
np.save('Reference_traj_fig8/coeffyl_5',coeffy5)
np.save('Reference_traj_fig8/coeffyl_6',coeffy6)
np.save('Reference_traj_fig8/coeffyl_7',coeffy7)
np.save('Reference_traj_fig8/coeffyl_8',coeffy8)

np.save('Reference_traj_fig8/coeffzl_1',coeffz1)
np.save('Reference_traj_fig8/coeffzl_2',coeffz2)
np.save('Reference_traj_fig8/coeffzl_3',coeffz3)
np.save('Reference_traj_fig8/coeffzl_4',coeffz4)
np.save('Reference_traj_fig8/coeffzl_5',coeffz5)
np.save('Reference_traj_fig8/coeffzl_6',coeffz6)
np.save('Reference_traj_fig8/coeffzl_7',coeffz7)
np.save('Reference_traj_fig8/coeffzl_8',coeffz8)


