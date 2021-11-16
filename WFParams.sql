create table param_set
(
	id INTEGER
		constraint param_set_pk
			primary key autoincrement,
	PE_per_MeV FLOAT default 1200 not null,
	gain FLOAT default -5000,
	PMT_sigma_t FLOAT default 3,
	lambda FLOAT default 1775,
	n FLOAT default 1.6,
	PMT_decay_proportion_1 FLOAT default 0.6,
	PMT_decay_proportion_2 FLOAT default 0.4,
	PMT_decay_tau_1 FLOAT default 0.5,
	PMT_decay_tau_2 FLOAT default 16,
	PSD_response_1_p1 FLOAT default 0.7,
	PSD_response_1_p2 FLOAT default 0.28,
	PSD_response_1_p3 FLOAT default 0.02,
	PSD_response_1_tau1 FLOAT default 3.16,
	PSD_response_1_tau2 FLOAT default 32.3,
	PSD_response_1_tau3 FLOAT default 270,
	PSD_response_2_p1 FLOAT default 0.3,
	PSD_response_2_p2 FLOAT default 0.65,
	PSD_response_2_p3 FLOAT default 0.05,
	PSD_response_2_tau1 FLOAT default 3.16,
	PSD_response_2_tau2 FLOAT default 32.3,
	PSD_response_2_tau3 FLOAT default 270,
	name STRING UNIQUE not null,
	zoff FLOAT default 0
);
