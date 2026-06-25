// Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
// This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
// See LICENSE for details.

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(ElementPlus)

app.mount('#app')
